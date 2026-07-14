"""Re-synthesise the flagged transcriptions with Coqui TTS and re-transcribe.

The audio-quality check (``scripts/verify_audio.py``) flags trials whose
StyleTTS2 audio round-trips poorly through a Whisper ASR model. A high flag can
mean two very different things:

* the TTS glitched on that clip (a different engine would say it cleanly), or
* the text itself is just hard for ASR (word lists, ``[RIGHT HAND - CLOSE]``,
  single tokens like ``bah`` — Whisper spells them out or mishears them).

This script takes the flagged ``ref`` texts, synthesises each one afresh with an
INDEPENDENT engine (Coqui TTS), transcribes that new audio with Whisper, and
scores it against the ref. Comparing the Coqui round-trip WER to the original
StyleTTS2 one tells you which bucket a flag falls in: if Coqui also round-trips
badly, the text is the problem, not StyleTTS2.

Reads settings from the ``recheck_audio`` section of the config, or run directly:
    python scripts/recheck_flagged.py                       # config defaults
    python scripts/recheck_flagged.py --flagged-csv path.csv --limit 50
    python scripts/recheck_flagged.py --no-dedupe           # one row per flag

Requires Coqui TTS:  pip install TTS
"""
import argparse
import csv
import os
import sys

import numpy as np

# Make the project root importable when run as `python scripts/recheck_flagged.py`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training.metrics import word_error_rate
from src.utils.config import resolve_device
from src.utils.logging_utils import get_logger
from src.utils.text import english_normalizer

logger = get_logger("recheck_flagged")

WHISPER_SR = 16000      # sample rate Whisper expects


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--flagged-csv", default="results/audio_quality/styletts2/tiny.en/tiny.en_train-val.csv",
                   help="flagged-items CSV from verify_audio.py (its `ref` column drives this)")
    p.add_argument("--tts-model", default="tts_models/en/ljspeech/tacotron2-DDC",
                   help="Coqui TTS model id to synthesise with")
    p.add_argument("--whisper-model", default="tiny.en",
                   help="Whisper model to transcribe the Coqui audio with")
    p.add_argument("--engine", default="styletts2",
                   help="which engine's flags are being rechecked (namespaces the output)")
    p.add_argument("--output-dir", default="results",
                   help="base dir; CSV -> <output_dir>/recheck/<engine>/<whisper_model>/<csv_stem>.csv")
    p.add_argument("--limit", type=int, default=0, help="cap number of texts (0 = all)")
    p.add_argument("--no-dedupe", dest="dedupe", action="store_false",
                   help="synthesise every flagged row (default: unique refs only)")
    p.add_argument("--device", default="auto", help="auto | cuda | cpu")
    return p.parse_args()


def load_refs(csv_path, dedupe):
    """Read flagged rows; return list of (ref, orig_asr, orig_wer) with a ref.

    With ``dedupe`` (default), each distinct ref appears once — many flags repeat
    the same short prompt (e.g. ``[DO NOTHING]``), so this avoids re-synthesising
    identical text. The kept row is the worst (highest original WER) for that ref.
    """
    rows = []
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            ref = (r.get("ref") or "").strip()
            if not ref:
                continue
            try:
                orig_wer = float(r.get("wer") or "nan")
            except ValueError:
                orig_wer = float("nan")
            rows.append((ref, (r.get("asr") or "").strip(), orig_wer))

    if not dedupe:
        return rows

    best = {}
    for ref, asr, wer in rows:
        prev = best.get(ref)
        if prev is None or (wer == wer and wer > prev[2]):   # wer==wer skips NaN
            best[ref] = (ref, asr, wer)
    return list(best.values())


def run(args):
    """Re-synthesise flagged texts with Coqui TTS and re-transcribe with Whisper.

    ``args`` is an argparse Namespace or a config SimpleNamespace with the same
    attribute names (``flagged_csv``, ``tts_model``, ``whisper_model``,
    ``output_dir``, ``limit``, ``dedupe``, ``device``).
    """
    args.device = resolve_device(args.device)
    normalize = english_normalizer()

    refs = load_refs(args.flagged_csv, args.dedupe)
    if args.limit:
        refs = refs[: args.limit]
    if not refs:
        logger.error(f"no usable `ref` rows in {args.flagged_csv}")
        return
    logger.info("=" * 70)
    logger.info(f"rechecking {len(refs)} text(s) from {args.flagged_csv} "
                f"(dedupe={args.dedupe})")

    # ── load the two independent models ──
    from TTS.api import TTS
    logger.info(f"loading Coqui TTS '{args.tts_model}' on {args.device}")
    tts = TTS(args.tts_model).to(args.device)
    tts_sr = tts.synthesizer.output_sample_rate

    import whisper
    logger.info(f"loading Whisper '{args.whisper_model}' on {args.device}")
    asr = whisper.load_model(args.whisper_model, device=args.device)

    import librosa

    out_rows, wers = [], []
    for i, (ref, orig_asr, orig_wer) in enumerate(refs):
        try:
            wav = np.asarray(tts.tts(text=ref), dtype=np.float32)
            if tts_sr != WHISPER_SR:
                wav = librosa.resample(wav, orig_sr=tts_sr, target_sr=WHISPER_SR)
            hyp = asr.transcribe(wav, language="en",
                                 fp16=(args.device == "cuda"))["text"].strip()
            wer = word_error_rate(normalize(ref), normalize(hyp))
        except Exception as e:
            logger.warning(f"failed on {ref!r}: {type(e).__name__}: {e}")
            hyp, wer = "", float("nan")

        wers.append(wer)
        delta = (wer - orig_wer) if (wer == wer and orig_wer == orig_wer) else float("nan")
        out_rows.append(dict(ref=ref, coqui_asr=hyp,
                             coqui_wer=round(wer, 4) if wer == wer else "",
                             styletts2_asr=orig_asr,
                             styletts2_wer=round(orig_wer, 4) if orig_wer == orig_wer else "",
                             delta_wer=round(delta, 4) if delta == delta else ""))
        tag = "  "
        if wer == wer and orig_wer == orig_wer:
            tag = "OK" if wer < orig_wer else "=="  # Coqui better / not better
        logger.info(f"{tag}[{i+1}/{len(refs)}] coqui-WER "
                    f"{wer:.2f} (was {orig_wer:.2f}) | ref: {ref!r} | coqui-asr: {hyp!r}")

    # ── output CSV ──
    engine = getattr(args, "engine", "styletts2")
    stem = os.path.splitext(os.path.basename(args.flagged_csv))[0]
    out_path = os.path.join(args.output_dir, "recheck", engine, args.whisper_model, f"{stem}.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fields = ["ref", "coqui_asr", "coqui_wer", "styletts2_asr", "styletts2_wer", "delta_wer"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)

    # ── summary ──
    valid = [w for w in wers if w == w]
    if valid:
        mean = sum(valid) / len(valid)
        clean = sum(w < 0.1 for w in valid)
        improved = sum(1 for r in out_rows if isinstance(r["delta_wer"], float) and r["delta_wer"] < 0)
        logger.info("=" * 70)
        logger.info(f"Coqui round-trip: mean WER {mean:.3f} | "
                    f"{clean}/{len(valid)} now clean (<0.1) | "
                    f"{improved} improved vs StyleTTS2")
        logger.info("  → flags where Coqui is ALSO bad are hard-text, not TTS glitches.")
    logger.info(f"wrote {len(out_rows)} row(s) -> {out_path}")


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
