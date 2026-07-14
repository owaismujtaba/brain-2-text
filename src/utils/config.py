"""Load the YAML config and resolve runtime values.

Every setting for every stage lives in ``config.yaml``. A stage receives its own
section as a small attribute-accessible object (``cfg.epochs``, ``cfg.lr`` ...).
"""
import os
from types import SimpleNamespace

import torch
import yaml

# Whisper encoder hidden size (the embedding dimension we predict) per model.
# n_ctx (1500), neural_dim and frame_samples are the same across these models;
# only this dimension and the model name change.
WHISPER_EMB_DIM = {
    "tiny.en": 384, "tiny": 384,
    "base.en": 512, "base": 512,
    "small.en": 768, "small": 768,
}


def emb_dim_for(model_name: str) -> int:
    """Embedding dimension produced by a given Whisper model's encoder."""
    if model_name not in WHISPER_EMB_DIM:
        raise ValueError(
            f"unknown Whisper model '{model_name}'; known: {sorted(WHISPER_EMB_DIM)}")
    return WHISPER_EMB_DIM[model_name]


def model_paths(base_features_dir: str, base_ckpt_dir: str, model_name: str) -> tuple:
    """Per-model feature and checkpoint directories, so the models never collide."""
    return (os.path.join(base_features_dir, model_name),
            os.path.join(base_ckpt_dir, model_name))


def load_config(path: str = "config.yaml") -> dict:
    """Read the whole config file into a plain dict."""
    with open(path) as f:
        return yaml.safe_load(f)


def stage_config(config: dict, stage: str) -> SimpleNamespace:
    """Return one stage's settings as an attribute-accessible namespace."""
    return SimpleNamespace(**dict(config.get(stage, {})))


def resolve_device(value: str) -> str:
    """Turn 'auto' into 'cuda' when a GPU is available, otherwise 'cpu'."""
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return value
