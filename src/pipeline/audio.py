"""Stage 1: synthesise speech audio from each trial's transcription.

Two engines are supported, selected by ``cfg.engine``:
    * ``styletts2`` – StyleTTS2 (diffusion, voice-cloning params)
    * ``coqui``     – Coqui TTS (``cfg.coqui_model``)

For every trial we read its transcription from the raw HDF5, run text-to-speech,
resample to 16 kHz (what Whisper expects), and write the waveform to a mirror
HDF5 under ``generated_audio/<engine>/<split>/<session>/``. Namespacing by engine
means both engines' audio can live side by side, so a later stage can point at
whichever set it wants.

The work is spread across a pool of worker processes. Each worker loads its own
TTS model once (``_init_worker``) and then synthesises trials on demand. The
output is resumable: trials already present in the output file are skipped.
"""
import functools
import multiprocessing
import os
import warnings
from pathlib import Path

if not hasattr(functools, "cache"):
    functools.cache = functools.lru_cache(maxsize=None)

import h5py
import librosa
import numpy as np
import torch

from ..utils.logging_utils import get_logger

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("COQUI_TOS_AGREED", "1")   # auto-accept Coqui model terms

logger = get_logger("audio")

TARGET_SR = 16000       # Whisper's input sample rate

_engine = "styletts2"   # which TTS engine this worker runs, set in _init_worker
_worker_tts = None      # per-worker TTS model, created in _init_worker
_src_sr = 24000         # engine's native output sample rate (resample from this)
_diffusion_steps = 5    # StyleTTS2 synthesis params, set in _init_worker from cfg
_embedding_scale = 1.0
_alpha = 0.3
_beta = 0.7
_target_voice_path = ""


def _init_worker(engine="styletts2", threads_per_worker=1, diffusion_steps=5,
                 embedding_scale=1.0, alpha=0.3, beta=0.7, target_voice_path="",
                 coqui_model="tts_models/en/ljspeech/tacotron2-DDC"):
    global _engine, _worker_tts, _src_sr
    global _diffusion_steps, _embedding_scale, _alpha, _beta, _target_voice_path
    torch.set_num_threads(threads_per_worker)
    _engine = engine
    if engine == "styletts2":
        from styletts2 import tts
        logger.info(f"worker {os.getpid()} initialising StyleTTS2 "
                    f"(GPU: {torch.cuda.is_available()})")
        _worker_tts = tts.StyleTTS2()
        _src_sr = 24000
        _diffusion_steps = diffusion_steps
        _embedding_scale = embedding_scale
        _alpha = alpha
        _beta = beta
        _target_voice_path = target_voice_path
    elif engine == "coqui":
        from TTS.api import TTS
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"worker {os.getpid()} initialising Coqui TTS "
                    f"'{coqui_model}' on {device}")
        _worker_tts = TTS(coqui_model).to(device)
        _src_sr = _worker_tts.synthesizer.output_sample_rate
    else:
        raise ValueError(f"unknown engine '{engine}'; use 'styletts2' or 'coqui'")


def _synthesize(task):
    """Worker task: turn ``(trial, text)`` into ``(trial, text, audio_16k|None)``."""
    trial, text = task
    try:
        if _engine == "styletts2":
            audio = _worker_tts.inference(
                text=text, diffusion_steps=_diffusion_steps,
                embedding_scale=_embedding_scale, alpha=_alpha, beta=_beta,
                target_voice_path=(_target_voice_path or None), output_wav_file=None)
        else:  # coqui
            audio = np.asarray(_worker_tts.tts(text=text), dtype=np.float32)
        audio_16k = (audio if _src_sr == TARGET_SR
                     else librosa.resample(audio, orig_sr=_src_sr, target_sr=TARGET_SR))
    except Exception as e:
        logger.warning(f"TTS failed for '{trial}' (text: '{text}'): "
                       f"{type(e).__name__}: {e}", exc_info=True)
        return trial, text, None
    return trial, text, audio_16k


def _transcription(trial_group):
    """Extract a clean transcription string, or ``None`` if there isn't one."""
    if "transcription" not in trial_group:
        return None
    raw = trial_group["transcription"][()]
    if isinstance(raw, bytes):
        text = raw.decode("utf-8")
    elif isinstance(raw, np.ndarray):
        text = "".join(chr(c) for c in raw if c != 0)
    else:
        text = str(raw)
    text = text.strip()
    return text or None


def _pending_trials(raw_file, out_file):
    """List ``(trial, text)`` pairs that still need audio for one session."""
    existing = set()
    if os.path.exists(out_file):
        with h5py.File(out_file, "r") as done:
            existing = set(done.keys())

    pending = []
    with h5py.File(raw_file, "r") as raw:
        for trial in raw.keys():
            if trial in existing:
                continue
            text = _transcription(raw[trial])
            if text is None:
                logger.warning(f"trial '{trial}' has no transcription; skipping")
                continue
            pending.append((trial, text))
    return pending


def _process_session(raw_file, output_dir, split, pool):
    """Generate and store any missing audio for a single session file."""
    session = raw_file.parent.name
    session_dir = os.path.join(output_dir, split, session)
    os.makedirs(session_dir, exist_ok=True)
    out_file = os.path.join(session_dir, f"audio_{split}.hdf5")

    pending = _pending_trials(str(raw_file), out_file)
    if not pending:
        logger.info(f"[{session}] nothing to generate")
        return

    total, failed = len(pending), 0
    logger.info(f"[{session}] generating {total} trial(s)")
    with h5py.File(out_file, "a") as out:
        for done, (trial, text, audio_16k) in enumerate(
                pool.imap_unordered(_synthesize, pending), start=1):
            if audio_16k is None:
                failed += 1
                continue
            if trial in out:
                del out[trial]
            dset = out.create_dataset(trial, data=audio_16k, dtype="float32")
            dset.attrs["sentence_label"] = text
            out.flush()
            logger.info(f"[{session}] {done}/{total} '{trial}'")

    if failed:
        logger.warning(f"[{session}] {failed}/{total} trial(s) failed TTS")


def generate_audio(cfg):
    """Stage entry point: synthesise audio for every session in a split."""
    data_dir = Path(cfg.data_dir)
    split = cfg.split
    engine = getattr(cfg, "engine", "styletts2")
    coqui_model = getattr(cfg, "coqui_model", "tts_models/en/ljspeech/tacotron2-DDC")
    search = data_dir / split if (data_dir / split).exists() else data_dir
    raw_files = sorted(search.rglob(f"data_{split}.hdf5")) or sorted(search.rglob("*.hdf5"))

    # Namespace by engine so both engines' audio coexist and a later stage can
    # point its audio_dir at whichever one it wants.
    output_dir = os.path.join(cfg.output_dir, engine)

    logger.info(f"audio generation | engine={engine} | split={split} "
                f"| {len(raw_files)} session file(s) | {cfg.num_workers} workers "
                f"| out={output_dir}")

    ctx = multiprocessing.get_context("spawn")
    pool = ctx.Pool(processes=cfg.num_workers, initializer=_init_worker,
                    initargs=(engine, cfg.threads_per_worker, cfg.diffusion_steps,
                              cfg.embedding_scale, cfg.alpha, cfg.beta,
                              cfg.target_voice_path, coqui_model))
    try:
        for raw_file in raw_files:
            _process_session(raw_file, output_dir, split, pool)
    finally:
        pool.close()
        pool.join()
