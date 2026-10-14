"""Seed control."""
import os
import random

import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Scatter/pool ops on GPU remain nondeterministic; we report across-seed spread
    # rather than forcing torch.use_deterministic_algorithms (which breaks some PyG ops).
    os.environ.setdefault('PYTHONHASHSEED', str(seed))
