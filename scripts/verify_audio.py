"""Validate the StyleTTS2-generated audio against the source transcriptions.

Two kinds of check per trial:

* Structural (cheap, always run) – is the waveform present, finite, non-silent,
  and a sane duration? Catches dropped trials and TTS clips that came out empty,
  truncated, or as a runaway.
* Round-trip WER (needs Whisper) – transcribe the generated audio with an
  INDEPENDENT Whisper ASR model and compare to the stored ``sentence_label``.
  A WER near 0 means the TTS said the right words intelligibly; a high WER
  means wrong text, garbled speech, or corrupt audio.

Run from the project root:
    python scripts/verify_audio.py                       # all sessions, val split
    python scripts/verify_audio.py --split train
    python scripts/verify_audio.py --session t15.2023.10.13 --per-trial
    python scripts/verify_audio.py --structural-only      # skip Whisper (fast)
    python scripts/verify_audio.py --limit 20             # cap trials/session
"""
import argparse
import glob
import os
import sys

import h5py
import numpy as np

# Make the project root importable when run as `python scripts/verify_audio.py`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training.metrics import word_error_rate
from src.utils.config import resolve_device
from src.utils.logging_utils import get_logger
from src.utils.text import english_normalizer

logger = get_logger("verify_audio")

SR = 16000              # generated audio sample rate
MIN_SECS = 0.30         # shorter than this is almost certainly a TTS failure
MAX_SECS = 30.0         # Whisper's window; longer suggests a runaway synthesis
SILENCE_RMS = 1e-3      # below this the clip is effectively silent


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--audio-dir", default="data/generated_audio",
                   help="base dir; the <engine> subfolder holds <split>/<session>/audio_<split>.hdf5")
    p.add_argument("--engine", default="styletts2",
                   help="TTS engine whose audio to verify (styletts2 | coqui)")
    p.add_argument("--raw-dir", default="data/raw/hdf5_data_final",
                   help="raw dir (for the missing-trial count check)")
    p.add_argument("--split", nargs="+", default=["val"],
                   help="split(s) to verify: val | train | all | val train (default: val)")
    p.add_argument("--output-dir", default="results",
                   help="base dir for the flagged-items CSV; '' to skip. Written to "
                        "<output_dir>/audio_quality/<model>/<model>_<split>.csv")
    p.add_argument("--wer-flag", type=float, default=0.5,
                   help="round-trip WER at or above which a trial is flagged (default 0.5)")
    p.add_argument("--session", default="", help="verify only this session (default: all)")
    p.add_argument("--model", default="base.en",
                   help="Whisper ASR model for the round-trip check (independent of our pipeline)")
    p.add_argument("--limit", type=int, default=0, help="cap trials per session (0 = all)")
    p.add_argument("--structural-only", action="store_true",
                   help="skip the Whisper round-trip; run only structural checks")
    p.add_argument("--per-trial", action="store_true",
                   help="log every trial, not just the flagged ones")
    p.add_argument("--device", default="auto", help="auto | cuda | cpu")
    p.add_argument("--num-workers", type=int, default=1,
                   help="parallel worker processes across sessions (1 = serial)")
    p.add_argument("--threads-per-worker", type=int, default=0,
                   help="torch threads per worker (0 = leave default)")
    return p.parse_args()


def structural_flags(wav):
    """Return a list of structural problems with a waveform (empty = clean)."""
    flags = []
    if wav.size == 0:
        return ["empty"]
    if not np.all(np.isfinite(wav)):
        flags.append("non-finite")
    secs = wav.size / SR
    if secs < MIN_SECS:
        flags.append(f"too-short({secs:.2f}s)")
    elif secs > MAX_SECS:
        flags.append(f"too-long({secs:.1f}s)")
    rms = float(np.sqrt(np.mean(np.square(wav)))) if np.all(np.isfinite(wav)) else 0.0
    if rms < SILENCE_RMS:
        flags.append(f"silent(rms={rms:.1e})")
    return flags


def verify_session(audio_file, session, split, args, asr, normalize):
    """Verify one session file; return ``(summary, flag_rows)``.

    ``flag_rows`` is a list of dicts (one per flagged item) ready for the CSV.
    """
    flag_rows = []
    with h5py.File(audio_file, "r") as h:
        keys = sorted(h.keys())
        n_audio = len(keys)
        audio_trials = set(keys)
        if args.limit:
            keys = keys[: args.limit]

        wers, struct_bad, transcribed = [], 0, 0
        for k in keys:
            wav = np.asarray(h[k][()], dtype=np.float32)
            text = h[k].attrs.get("sentence_label", "")
            flags = structural_flags(wav)
            if flags:
                struct_bad += 1
                logger.warning(f"  [{split}/{session}/{k}] {', '.join(flags)} | ref: {text!r}")
                flag_rows.append(dict(split=split, session=session, trial=k,
                                      kind="structural:" + ";".join(flags),
                                      wer="", ref=text, asr=""))

            wer = None
            if asr is not None and "empty" not in flags and "non-finite" not in flags:
                hyp = asr.transcribe(wav, language="en",
                                     fp16=(args.device == "cuda"))["text"].strip()
                wer = word_error_rate(normalize(text), normalize(hyp))
                wers.append(wer)
                transcribed += 1
                if args.per_trial or wer >= args.wer_flag:
                    tag = "  " if wer < args.wer_flag else "!!"
                    logger.info(f"{tag}[{split}/{session}/{k}] WER {wer:.2f} | ref: {text!r} "
                                f"| asr: {hyp!r}")
                if wer >= args.wer_flag:
                    flag_rows.append(dict(split=split, session=session, trial=k,
                                          kind="high_wer", wer=round(wer, 4),
                                          ref=text, asr=hyp))

    # Missing trials: present in the raw file but never synthesised.
    raw_n, missing = None, None
    raw_file = os.path.join(args.raw_dir, session, f"data_{split}.hdf5")
    if os.path.exists(raw_file):
        with h5py.File(raw_file, "r") as raw:
            raw_keys = set(raw.keys())
        raw_n = len(raw_keys)
        for k in sorted(raw_keys - audio_trials):
            flag_rows.append(dict(split=split, session=session, trial=k,
                                  kind="missing", wer="", ref="", asr=""))
        missing = raw_n - n_audio

    mean_wer = (sum(wers) / len(wers)) if wers else None
    exact = (sum(w == 0.0 for w in wers) / len(wers)) if wers else None
    summary = dict(split=split, session=session, n_audio=n_audio, raw_n=raw_n,
                   missing=missing, struct_bad=struct_bad, transcribed=transcribed,
                   mean_wer=mean_wer, exact=exact)
    return summary, flag_rows


def write_report(path, flag_rows):
    """Write one CSV row per flagged item, sorted so the worst surface first."""
    import csv
    order = {"missing": 0, "structural": 1, "high_wer": 2}
    rows = sorted(flag_rows, key=lambda r: (
        order.get(r["kind"].split(":")[0], 9),
        -(r["wer"] if isinstance(r["wer"], float) else 0.0),
        r["split"], r["session"], r["trial"]))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["split", "session", "trial", "kind",
                                          "wer", "ref", "asr"])
        w.writeheader()
        w.writerows(rows)


# ── parallel workers ──
# Each worker process loads its OWN Whisper model once (in the initializer) and
# reuses it for every session it's handed, so the model load cost is paid
# num_workers times, not once per session.
_WORKER = {}


def _init_worker(model_name, device, threads_per_worker, structural_only):
    """Per-process setup: cap thread count and load one Whisper ASR model."""
    if threads_per_worker:
        import torch
        torch.set_num_threads(int(threads_per_worker))
    asr = None
    if not structural_only:
        import whisper
        asr = whisper.load_model(model_name, device=device)
    _WORKER["asr"] = asr
    _WORKER["normalize"] = english_normalizer()


def _verify_session_job(payload):
    """Pool task: verify one session using this worker's cached model."""
    audio_file, session, split, args = payload
    return verify_session(audio_file, session, split, args,
                          _WORKER["asr"], _WORKER["normalize"])


def _collect_jobs(args, splits):
    """Return the list of (audio_file, session, split) sessions to verify."""
    jobs = []
    for split in splits:
        pat = os.path.join(args.audio_dir, split, args.session or "*",
                           f"audio_{split}.hdf5")
        audio_files = sorted(glob.glob(pat))
        if not audio_files:
            logger.warning(f"no audio files match {pat}")
            continue
        logger.info(f"[{split}] {len(audio_files)} session(s)")
        for audio_file in audio_files:
            session = os.path.basename(os.path.dirname(audio_file))
            jobs.append((audio_file, session, split, args))
    return jobs


def run(args):
    """Verify generated audio using settings from ``args`` (argparse Namespace
    or a config SimpleNamespace with the same attribute names)."""
    args.device = resolve_device(args.device)
    # `split` may be a single value ("val"), the keyword "all", or a list
    # (["val", "train"]) from either the CLI or the config. "all" expands to both.
    requested = args.split if isinstance(args.split, (list, tuple)) else [args.split]
    splits = ["train", "val"] if "all" in requested else list(requested)

    num_workers = int(getattr(args, "num_workers", 1) or 1)
    threads_per_worker = int(getattr(args, "threads_per_worker", 0) or 0)

    # Read the chosen engine's audio (audio_dir/<engine>/<split>/...), matching
    # how generate_audio namespaces its output.
    args.engine = getattr(args, "engine", "styletts2")
    args.audio_dir = os.path.join(args.audio_dir, args.engine)

    logger.info("=" * 70)
    logger.info(f"verifying engine={args.engine} | split(s)={splits} | round-trip="
                f"{'off' if args.structural_only else args.model} "
                f"| wer-flag>={args.wer_flag} | workers={num_workers}")

    jobs = _collect_jobs(args, splits)
    if not jobs:
        logger.error("nothing verified; check --audio-dir/--split/--session")
        return

    summaries, flag_rows = [], []
    if num_workers > 1 and len(jobs) > 1:
        # Fan sessions out across processes; each loads its own model once.
        # `spawn` is required so CUDA can be initialised inside the children.
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(
                max_workers=min(num_workers, len(jobs)), mp_context=ctx,
                initializer=_init_worker,
                initargs=(args.model, args.device, threads_per_worker,
                          args.structural_only)) as ex:
            futures = [ex.submit(_verify_session_job, job) for job in jobs]
            for fut in as_completed(futures):
                summary, rows = fut.result()
                summaries.append(summary)
                flag_rows.extend(rows)
    else:
        # Serial path: load the model once in this process.
        _init_worker(args.model, args.device, threads_per_worker, args.structural_only)
        for job in jobs:
            summary, rows = _verify_session_job(job)
            summaries.append(summary)
            flag_rows.extend(rows)

    if not summaries:
        logger.error("nothing verified; check --audio-dir/--split/--session")
        return

    # ── overall report ──
    logger.info("=" * 70)
    total_missing = sum(s["missing"] or 0 for s in summaries)
    total_struct = sum(s["struct_bad"] for s in summaries)
    total_highwer = sum(1 for r in flag_rows if r["kind"] == "high_wer")
    all_wers = [s["mean_wer"] for s in summaries if s["mean_wer"] is not None]
    logger.info(f"sessions: {len(summaries)} | structural-bad: {total_struct} | "
                f"missing: {total_missing} | high-WER(>={args.wer_flag}): {total_highwer}")
    overall = None
    if all_wers:
        overall = sum(all_wers) / len(all_wers)
        logger.info(f"mean round-trip WER (session-averaged): {overall:.3f}")
        worst = sorted((s for s in summaries if s["mean_wer"] is not None),
                       key=lambda s: -s["mean_wer"])[:5]
        logger.info("worst 5 sessions by round-trip WER:")
        for s in worst:
            logger.info(f"  {s['split']}/{s['session']}  WER={s['mean_wer']:.3f}  "
                        f"exact={s['exact'] * 100:.0f}%  n={s['transcribed']}")

    if args.output_dir:
        # Namespace the report by the ASR model used, so runs with different
        # Whisper models don't overwrite each other's flags.
        split_tag = "-".join(splits)
        report_path = os.path.join(args.output_dir, "audio_quality", args.engine,
                                   args.model, f"{args.model}_{split_tag}.csv")
        write_report(report_path, flag_rows)
        logger.info(f"wrote {len(flag_rows)} flagged item(s) -> {report_path}")

    clean = total_struct == 0 and total_missing == 0 and (overall is None or overall < 0.1)
    logger.info("RESULT: audio looks good." if clean
                else "RESULT: review the flagged items in the report above.")


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
