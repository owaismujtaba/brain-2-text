"""Turn predicted embeddings into text via the frozen Whisper decoder, and score.

Two entry points:
    * ``decode_dataset`` – decode trials of a dataset; returns (WER, exact, rows).
      Shared by the training loop's periodic WER check.
    * ``decode_split``   – the ``python main.py decode`` stage: load a checkpoint,
      decode a whole split, and write a per-trial CSV.
"""
import csv
import os
import time
from collections import defaultdict

import torch

from ..data.dataset import NeuralEmbeddingDataset
from ..model.model import ConvBiGRU
from ..utils.logging_utils import get_logger
from ..utils.text import collapse_runaway_repeats, english_normalizer
from .metrics import word_error_rate

logger = get_logger("decode")


@torch.no_grad()
def decode_dataset(model, dataset, device, whisper_model=None, model_name="tiny.en",
                   beam_size=5, limit=0, n_ctx=1500, emb_dim=384, verbose=False):
    """Decode trials and return ``(mean_wer, exact_match_fraction, rows)``.

    ``rows`` is a list of ``{"idx", "wer", "truth", "pred"}`` dicts.

    When ``verbose`` is set, log progress (running WER and decoding rate) every
    ~10% of trials and warn on every repetition-loop collapse. Left off for the
    training loop's periodic WER check so it doesn't flood ``train.log``.
    """
    import whisper
    if whisper_model is None:
        whisper_model = whisper.load_model(model_name, device=device)
    options = whisper.DecodingOptions(
        language="en", without_timestamps=True,
        fp16=(str(device) == "cuda"),
        beam_size=(beam_size if beam_size and beam_size > 1 else None),
    )
    normalize_text = english_normalizer()
    model.eval()

    count = len(dataset) if not limit else min(limit, len(dataset))
    rows, errors, exact, collapsed = [], [], 0, 0
    start = time.time()
    log_every = max(1, count // 10)
    for i in range(count):
        neural, target, truth = dataset[i]
        n_frames = target.shape[0]
        session, trial, _ = dataset.index[i]

        # Only the first n_frames query frames are used, so compute just those:
        # cross-attention treats each query independently, so this matches slicing
        # pred[:n_frames] out of a full n_ctx forward but skips the unused frames.
        pred = model(neural.unsqueeze(0).to(device), n_out=n_frames)[0]
        feats = torch.zeros(n_ctx, emb_dim, device=device)
        feats[:n_frames] = pred                                # keep content frames only
        result = whisper.decode(whisper_model, feats.unsqueeze(0).float(), options)
        raw_text = (result[0] if isinstance(result, list) else result).text.strip()
        text = collapse_runaway_repeats(raw_text)
        if text != raw_text:
            collapsed += 1
            if verbose:
                logger.info(f"  [guard] collapsed repetition loop at trial {i} "
                            f"({session}/{trial}): {raw_text!r} -> {text!r}")

        ref, hyp = normalize_text(truth), normalize_text(text)
        error = word_error_rate(ref, hyp)
        errors.append(error)
        exact += int(ref == hyp)
        rows.append({"idx": i, "session": session, "trial": trial,
                     "wer": round(error, 4), "truth": truth, "pred": text})

        if verbose and ((i + 1) % log_every == 0 or i + 1 == count):
            elapsed = time.time() - start
            rate = (i + 1) / elapsed if elapsed > 0 else 0.0
            eta = (count - (i + 1)) / rate if rate > 0 else 0.0
            logger.info(f"  progress {i + 1}/{count} ({100 * (i + 1) / count:.0f}%) "
                        f"| running WER {sum(errors) / len(errors):.4f} "
                        f"| {rate:.1f} trials/s | ETA {eta / 60:.1f} min")

    mean_wer = sum(errors) / max(len(errors), 1)
    if verbose:
        logger.info(f"decoded {count} trials in {(time.time() - start) / 60:.1f} min "
                    f"| repetition-loops collapsed: {collapsed}")
    return mean_wer, exact / max(len(errors), 1), rows


def _decode_run_name(cfg):
    """Rebuild the checkpoint folder name from the decode config, matching the
    exact convention training uses (`train._run_name`), so the decode section
    alone points at the run to evaluate."""
    return (f"{cfg.model}_c{cfg.conv_channels}_h{cfg.hidden}_g{cfg.gru_layers}"
            f"_norm{int(cfg.normalize)}_aug{int(cfg.augment)}"
            f"_dec{cfg.dec_loss_weight:g}")


def decode_split(cfg):
    """Stage: decode a whole split with a saved checkpoint and write a CSV.

    Which model/run to decode is driven entirely by the `decode` section of the
    config: `model` plus the architecture/loss fields rebuild the checkpoint's
    run-folder name, and `ckpt_dir`/`ckpt_name` locate the file inside it.
    """
    device = cfg.device

    # The Whisper model to decode with comes from the decode config, not the
    # checkpoint. It also selects the per-model feature folder below.
    model_name = cfg.model

    # Which TTS engine's features to decode against (must match how the
    # checkpoint was trained). Namespaces both the feature folder and the output.
    engine = getattr(cfg, "engine", "styletts2")

    # Locate the checkpoint from the decode config alone:
    #   <ckpt_dir>/<engine>/<run_name>/<ckpt_name>
    run_name = _decode_run_name(cfg)
    cfg.ckpt = os.path.join(cfg.ckpt_dir, engine, run_name, cfg.ckpt_name)

    ckpt = torch.load(cfg.ckpt, map_location=device, weights_only=False)
    saved = ckpt["args"]      # the training config, so we rebuild the exact model

    n_ctx = int(saved.get("n_ctx", 1500))
    emb_dim = int(saved.get("emb_dim", 384))
    neural_dim = int(saved.get("neural_dim", 512))
    frame_samples = int(saved.get("frame_samples", 320))
    features_dir = os.path.join(cfg.features_dir, engine, model_name)

    logger.info("=" * 60)
    logger.info(f"Decoding split='{cfg.split}' with checkpoint {cfg.ckpt}")
    logger.info(f"model={model_name} | emb_dim={emb_dim} | beam_size={cfg.beam_size} "
                f"| features={features_dir} | device={device}")

    model = ConvBiGRU(
        neural_dim=neural_dim, conv_channels=saved.get("conv_channels", 256),
        hidden=saved["hidden"], gru_layers=saved["gru_layers"],
        emb_dim=emb_dim, n_ctx=n_ctx, dropout=saved["dropout"]).to(device)
    model.load_state_dict(ckpt["model"])
    logger.info(f"loaded model (epoch {ckpt.get('epoch')}, gru_layers {saved['gru_layers']})")

    normalize = bool(saved.get("normalize", True))   # must match how it was trained
    dataset = NeuralEmbeddingDataset(
        cfg.raw_dir, features_dir, cfg.split,
        normalize=normalize, augment=None,
        n_ctx=n_ctx, neural_dim=neural_dim, frame_samples=frame_samples)
    logger.info(f"{cfg.split} trials: {len(dataset)} "
                f"(decoding {cfg.limit or len(dataset)}) | normalize={normalize}")

    mean_wer, exact, rows = decode_dataset(
        model, dataset, device, model_name=model_name,
        beam_size=cfg.beam_size, limit=cfg.limit, n_ctx=n_ctx, emb_dim=emb_dim,
        verbose=True)

    # Namespace the output by the run folder (reusing the name we resolved the
    # checkpoint from) so different runs never overwrite each other, and name the
    # file by the model and beam size used so decode variants stay distinct.
    fname = f"{model_name}_beam{cfg.beam_size}_{cfg.split}.csv"
    out_path = os.path.join(cfg.output_dir, "decode", engine, run_name, fname)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["idx", "session", "trial", "wer", "truth", "pred"])
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"WER {mean_wer:.4f} | exact {exact * 100:.1f}% "
                f"| wrote {len(rows)} rows -> {out_path}")

    _log_wer_summary(rows, mean_wer)
    return mean_wer, exact


def _log_wer_summary(rows, mean_wer):
    """Log distribution, per-session breakdown, and sample predictions."""
    wers = sorted(row["wer"] for row in rows)
    n = len(wers)
    if n:
        median = wers[n // 2]
        perfect = sum(w == 0.0 for w in wers)
        runaway = sum(w > 1.0 for w in wers)
        p90 = wers[min(n - 1, int(0.90 * n))]
        logger.info(f"  WER stats: mean {mean_wer:.4f} | median {median:.4f} | "
                    f"p90 {p90:.4f} | worst {wers[-1]:.2f}")
        logger.info(f"  trials: {perfect} perfect (WER=0, {100 * perfect / n:.1f}%) | "
                    f"{runaway} with WER>1 (likely word-list/hard trials)")

        # WER distribution buckets
        edges = [0.0, 0.1, 0.25, 0.5, 1.0, float("inf")]
        labels = ["=0", "0-0.1", "0.1-0.25", "0.25-0.5", "0.5-1.0", ">1.0"]
        counts = [0] * len(labels)
        for w in wers:
            if w == 0.0:
                counts[0] += 1
            else:
                for b in range(1, len(edges)):
                    if w <= edges[b]:
                        counts[b] += 1
                        break
        logger.info("  distribution: " + " | ".join(
            f"{lab} {c} ({100 * c / n:.0f}%)" for lab, c in zip(labels, counts)))

        # per-session WER, worst 5 first
        by_session = defaultdict(list)
        for row in rows:
            by_session[row["session"]].append(row["wer"])
        sess = sorted(((s, sum(v) / len(v), len(v)) for s, v in by_session.items()),
                      key=lambda t: -t[1])
        logger.info(f"  {len(sess)} sessions | best {sess[-1][0]}={sess[-1][1]:.3f} "
                    f"| worst {sess[0][0]}={sess[0][1]:.3f}")
        logger.info("  worst 5 sessions:")
        for s, w, k in sess[:5]:
            logger.info(f"    {s}  n={k}  WER={w:.3f}")

    logger.info("  sample predictions:")
    for row in rows[:8]:
        logger.info(f"  [{row['wer']:.2f}] truth: {row['truth']}")
        logger.info(f"          pred : {row['pred']}")
