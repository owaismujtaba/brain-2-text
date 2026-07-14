"""Reproducible seeding for training and inference.

``seed_everything`` fixes the Python, NumPy and PyTorch (CPU + CUDA) RNGs so a
run can be reproduced. ``seed_worker`` gives each DataLoader worker a distinct
but deterministic seed, and ``make_generator`` returns a seeded generator for
reproducible shuffling. Set ``deterministic=True`` for bit-exact CUDA kernels
(slower); the default keeps cuDNN fast but still seeds every RNG.
"""
import os
import random

import numpy as np
import torch


def seed_everything(seed=42, deterministic=False):
    """Seed all RNGs. Returns the seed so callers can log it."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed


def seed_worker(worker_id):
    """DataLoader ``worker_init_fn``: derive each worker's seed from the base seed."""
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed=42):
    """A seeded ``torch.Generator`` for reproducible DataLoader shuffling."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g
