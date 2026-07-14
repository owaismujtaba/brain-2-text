"""Train the ConvBiGRU to map neural activity to Whisper encoder embeddings.

The loop is deliberately plain and evaluates on the *entire* validation split:

    for each epoch:
        train pass  – one pass over ALL training trials (updates weights)
        val pass    – one pass over ALL validation trials (no updates)
        step the learning-rate schedule
        select best.pt by full-val loss; every `wer_every` epochs also decode
        the full val set to track best_wer.pt; always write last.pt

Checkpoint selection uses the teacher-forced loss over the full validation set
(cheap, every epoch). Because loss is only a proxy for what we ultimately care
about, a free-running Whisper WER decode over the full val set runs every
`wer_every` epochs and keeps a separate best-WER checkpoint.

Two losses can be combined (see ``losses.py``):
    * an embedding regression loss (always on), and
    * an optional decoder-in-the-loop loss that scores decodability directly.
"""
import csv
import os
from functools import partial

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data.dataset import NeuralEmbeddingDataset, collate, NeuralAugment
from ..model.model import ConvBiGRU
from ..model.losses import embedding_loss, WhisperDecoderLoss
from ..utils.config import emb_dim_for
from ..utils.logging_utils import get_logger
from ..utils.seed import seed_everything, seed_worker, make_generator
from .decode import decode_dataset

logger = get_logger("train")


def describe_device(device):
    """Human-readable description of the training device."""
    if str(device).startswith("cuda") and torch.cuda.is_available():
        idx = torch.cuda.current_device() if device == "cuda" else int(str(device).split(":")[-1])
        name = torch.cuda.get_device_name(idx)
        total = torch.cuda.get_device_properties(idx).total_memory / 1e9
        return f"CUDA (GPU {idx}: {name}, {total:.1f} GB, {torch.cuda.device_count()} visible)"
    return f"CPU ({torch.get_num_threads()} threads)"


def run_epoch(model, loader, device, optimizer=None, decoder_loss=None, dec_weight=0.0):
    """One full pass over ``loader``; returns mean ``[total, l1, cos, dec]`` losses.

    Passing an ``optimizer`` puts the model in train mode and updates weights;
    passing ``None`` runs a no-grad evaluation pass over the whole split.
    """
    training = optimizer is not None
    model.train() if training else model.eval()
    totals = torch.zeros(4)          # total, l1, cos, dec
    seen = 0

    non_blocking = (str(device) == "cuda")
    with torch.set_grad_enabled(training):
        for neural, lengths, target, mask, texts in tqdm(loader, leave=False):
            # lengths stays on CPU (pack_padded_sequence needs it there); the rest
            # copy asynchronously so the H2D transfer overlaps the previous step.
            neural = neural.to(device, non_blocking=non_blocking)
            target = target.to(device, non_blocking=non_blocking)
            mask = mask.to(device, non_blocking=non_blocking)

            # Only compute over the frames any item in the batch actually uses.
            valid = int(mask.sum(dim=1).max().item())
            pred = model(neural, lengths, n_out=valid)
            batch_mask = mask[:, :valid]
            loss, l1, cos = embedding_loss(pred, target[:, :valid], batch_mask)

            dec = torch.zeros((), device=device)
            if decoder_loss is not None and dec_weight > 0:
                dec = decoder_loss(pred, batch_mask, texts)
                loss = loss + dec_weight * dec

            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            batch_size = neural.size(0)
            totals += torch.tensor([loss.item(), l1.item(), cos.item(), float(dec)]) * batch_size
            seen += batch_size

    return (totals / max(seen, 1)).tolist()


def _build_datasets(cfg, features_dir, dims):
    """Create the train (augmented) and val (clean) datasets over the full splits."""
    augment = None
    if bool(getattr(cfg, "augment", True)):
        augment = NeuralAugment(
            noise_std=getattr(cfg, "aug_noise_std", 0.15),
            channel_drop=getattr(cfg, "aug_channel_drop", 0.1),
            scale=getattr(cfg, "aug_scale", 0.1),
            time_jitter=getattr(cfg, "aug_time_jitter", 2),
        )
    normalize = bool(getattr(cfg, "normalize", True))

    train_ds = NeuralEmbeddingDataset(
        cfg.raw_dir, features_dir, "train",
        normalize=normalize, augment=augment, **dims)
    val_ds = NeuralEmbeddingDataset(
        cfg.raw_dir, features_dir, "val",
        normalize=normalize, augment=None, **dims)

    # `limit` is a debugging aid only; 0 (the default) uses every trial.
    if cfg.limit:
        train_ds.index = train_ds.index[: cfg.limit]
        val_ds.index = val_ds.index[: cfg.limit]

    logger.info(f"neural input: normalize={normalize} | augment={augment is not None}")
    logger.info(f"train trials: {len(train_ds)} | val trials: {len(val_ds)} "
                f"(validating on the FULL val split)")
    return train_ds, val_ds


def _save(model, optimizer, scheduler, cfg, epoch, best_val, best_wer, name,
          epochs_no_improve=0, **extra):
    """Write a checkpoint that records everything needed to resume or rebuild."""
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "args": vars(cfg),
        "epoch": epoch,
        "best_val": best_val,
        "best_wer": best_wer,
        "epochs_no_improve": epochs_no_improve,
    }
    ckpt.update(extra)
    path = os.path.join(cfg.ckpt_dir, name)
    torch.save(ckpt, path)
    return path


def _write_predictions_csv(rows, path):
    """Write decode_dataset's per-trial rows as (session, trial, wer, actual, predicted)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["session", "trial", "wer", "actual", "predicted"])
        writer.writeheader()
        for row in rows:
            writer.writerow({"session": row["session"], "trial": row["trial"],
                             "wer": row["wer"], "actual": row["truth"], "predicted": row["pred"]})


def _resume(model, optimizer, scheduler, cfg, device):
    """Restore the most recent checkpoint if one exists; returns training state."""
    for name in ("last.pt", "best.pt"):
        path = os.path.join(cfg.ckpt_dir, name)
        if not os.path.exists(path):
            continue
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        # Optimizer/scheduler state is best-effort: if the param groups changed,
        # keep the restored model weights but start the optimizer fresh instead
        # of crashing.
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
        except (ValueError, KeyError) as e:
            logger.warning(f"could not restore optimizer/scheduler state ({e}); "
                           "continuing with a fresh optimizer")
        if scheduler.T_max != cfg.epochs:
            logger.info(f"fixing scheduler T_max: {scheduler.T_max} -> {cfg.epochs}")
            scheduler.T_max = cfg.epochs
        best_val = ckpt.get("best_val", float("inf"))
        best_wer = ckpt.get("best_wer", float("inf"))
        epochs_no_improve = ckpt.get("epochs_no_improve", 0)
        start_epoch = ckpt.get("epoch", 0) + 1
        logger.info(f"resumed from {path} (epoch {start_epoch - 1}, "
                    f"best val {best_val:.4f}, best wer {best_wer:.4f})")
        return start_epoch, best_val, best_wer, epochs_no_improve
    return 1, float("inf"), float("inf"), 0


def _run_name(cfg, model_name, conv_channels, hidden, gru_layers, normalize, augment,
              dec_loss_weight):
    """Checkpoint folder name so different targets/architectures never collide."""
    return (f"{model_name}_c{conv_channels}_h{hidden}_g{gru_layers}"
            f"_norm{int(normalize)}_aug{int(augment)}_dec{dec_loss_weight:g}")


def _full_val_wer(model, val_ds, device, model_name, decoder_loss, cfg, n_ctx, emb_dim):
    """Free-running Whisper WER over the ENTIRE validation split."""
    return decode_dataset(
        model, val_ds, device, model_name=model_name,
        whisper_model=(decoder_loss.whisper if decoder_loss else None),
        beam_size=getattr(cfg, "beam_size", 5),
        limit=0, n_ctx=n_ctx, emb_dim=emb_dim)


def train(cfg):
    """Full training run driven by the ``train`` config section."""
    device = cfg.device

    seed = int(getattr(cfg, "seed", 42))
    seed_everything(seed, deterministic=bool(getattr(cfg, "deterministic", False)))

    # Which Whisper target we're training. The embedding size, the feature folder
    # and the checkpoint folder all follow from this one choice, so the three
    # model sizes never collide and can't be mismatched.
    model_name = getattr(cfg, "model", "tiny.en")
    emb_dim = emb_dim_for(model_name)

    # Which TTS engine's features we train on. It namespaces both the feature
    # folder and the checkpoint folder, so runs on different engines' audio never
    # collide (and decode can match the pair via its own `engine`).
    engine = getattr(cfg, "engine", "styletts2")
    features_dir = os.path.join(cfg.features_dir, engine, model_name)

    conv_channels = int(getattr(cfg, "conv_channels", 256))
    hidden = int(cfg.hidden)
    gru_layers = int(cfg.gru_layers)
    normalize = bool(getattr(cfg, "normalize", True))
    augment = bool(getattr(cfg, "augment", True))
    dec_loss_weight = float(getattr(cfg, "dec_loss_weight", 0.0))
    run_name = _run_name(cfg, model_name, conv_channels, hidden, gru_layers,
                         normalize, augment, dec_loss_weight)
    ckpt_dir = os.path.join(cfg.ckpt_dir, engine, run_name)
    cfg.model, cfg.emb_dim, cfg.ckpt_dir = model_name, emb_dim, ckpt_dir
    os.makedirs(ckpt_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info(f"Training neural -> Whisper-embedding model | target '{model_name}'")
    logger.info(f"seed: {seed} | deterministic={bool(getattr(cfg, 'deterministic', False))}")
    logger.info(f"device: {device} -> {describe_device(device)}")
    logger.info(f"emb_dim={emb_dim} | features={features_dir} | ckpts={ckpt_dir}")
    logger.info(f"config: {vars(cfg)}")

    n_ctx = int(getattr(cfg, "n_ctx", 1500))
    neural_dim = int(getattr(cfg, "neural_dim", 512))
    frame_samples = int(getattr(cfg, "frame_samples", 320))
    dims = dict(n_ctx=n_ctx, neural_dim=neural_dim, frame_samples=frame_samples)

    train_ds, val_ds = _build_datasets(cfg, features_dir, dims)

    collate_fn = partial(collate, n_ctx=n_ctx)
    pin = (device == "cuda")
    # Keep the GPU fed: workers persist across epochs (no re-fork / HDF5 re-open
    # at every boundary) and each stays several batches ahead via prefetch.
    loader_kwargs = dict(num_workers=cfg.num_workers, collate_fn=collate_fn,
                         pin_memory=pin)
    if cfg.num_workers > 0:
        loader_kwargs.update(persistent_workers=True,
                             prefetch_factor=getattr(cfg, "prefetch_factor", 4),
                             worker_init_fn=seed_worker)
    # Seeded generator so the shuffle order is identical across reruns.
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              drop_last=True, generator=make_generator(seed), **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            **loader_kwargs)

    model = ConvBiGRU(
        neural_dim=neural_dim, conv_channels=conv_channels,
        hidden=hidden, gru_layers=gru_layers,
        emb_dim=emb_dim, n_ctx=n_ctx, dropout=cfg.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"model: ConvBiGRU | {n_params / 1e6:.2f}M params")

    decoder_loss = None
    if dec_loss_weight > 0:
        decoder_loss = WhisperDecoderLoss(model_name, device, n_ctx=n_ctx)
        logger.info(f"decoder-in-the-loop loss ENABLED | weight {dec_loss_weight}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    start_epoch, best_val, best_wer, epochs_no_improve = _resume(
        model, optimizer, scheduler, cfg, device)

    # Warm-start the encoder from another checkpoint on a *fresh* run. Only the
    # ConvBiGRU weights are copied; optimizer/epoch start clean.
    init_from = getattr(cfg, "init_from", "")
    if start_epoch == 1 and init_from:
        src = torch.load(init_from, map_location=device, weights_only=False)
        model.load_state_dict(src["model"])
        logger.info(f"warm-started encoder weights from {init_from} (epoch {src.get('epoch')})")

    patience = int(getattr(cfg, "patience", 0))
    wer_every = int(getattr(cfg, "wer_every", 0))
    logger.info("checkpoint selection: full-val loss every epoch (-> best.pt)"
                + (f"; full-val WER every {wer_every} epoch(s) (-> best_wer.pt)"
                   if wer_every else "; periodic WER disabled"))

    for epoch in range(start_epoch, cfg.epochs + 1):
        lr_now = scheduler.get_last_lr()[0]
        tr = run_epoch(model, train_loader, device, optimizer, decoder_loss, dec_loss_weight)
        va = run_epoch(model, val_loader, device, None, decoder_loss, dec_loss_weight)
        scheduler.step()

        dec_str = f" | dec {tr[3]:.4f}/{va[3]:.4f}" if dec_loss_weight > 0 else ""
        logger.info(f"epoch {epoch:3d}/{cfg.epochs} | lr {lr_now:.2e} "
                    f"| train {tr[0]:.4f} (l1 {tr[1]:.4f} cos {tr[2]:.4f}) "
                    f"| val {va[0]:.4f} (l1 {va[1]:.4f} cos {va[2]:.4f})" + dec_str)

        # ── Selection & early stopping: full-val teacher-forced loss ──
        val_loss = va[0]
        if val_loss < best_val:
            best_val = val_loss
            epochs_no_improve = 0
            path = _save(model, optimizer, scheduler, cfg, epoch, best_val, best_wer,
                         "best.pt", epochs_no_improve=0)
            logger.info(f"           new best val loss {best_val:.4f} -> {path}")
        else:
            epochs_no_improve += 1

        # ── Periodic true-metric check: WER over the FULL val split ──
        is_last = (epoch == cfg.epochs) or (patience and epochs_no_improve >= patience)
        if wer_every and (epoch % wer_every == 0 or is_last):
            mean_wer, exact, rows = _full_val_wer(
                model, val_ds, device, model_name, decoder_loss, cfg, n_ctx, emb_dim)
            logger.info(f"           full-val WER {mean_wer:.4f} | exact {exact * 100:.1f}% "
                        f"(over {len(val_ds)} val trials)")
            for row in rows[:3]:
                logger.info(f"             truth: {row['truth']}")
                logger.info(f"             pred : {row['pred']}")
            _write_predictions_csv(rows, os.path.join(cfg.ckpt_dir, "val_predictions.csv"))
            if mean_wer < best_wer:
                best_wer = mean_wer
                path = _save(model, optimizer, scheduler, cfg, epoch, best_val, best_wer,
                             "best_wer.pt", wer=mean_wer)
                _write_predictions_csv(rows, os.path.join(cfg.ckpt_dir, "best_predictions.csv"))
                logger.info(f"           new best full-val WER {best_wer:.4f} -> {path}")

        _save(model, optimizer, scheduler, cfg, epoch, best_val, best_wer, "last.pt",
              epochs_no_improve=epochs_no_improve)

        if patience and epochs_no_improve >= patience:
            logger.info(f"early stopping: no val-loss improvement for "
                        f"{epochs_no_improve} epochs (patience={patience})")
            break

    logger.info(f"done. best val loss {best_val:.4f} | best full-val WER {best_wer:.4f} "
                f"| checkpoints in {cfg.ckpt_dir}/")
    return best_val
