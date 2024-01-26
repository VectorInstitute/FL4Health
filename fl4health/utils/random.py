import random
from logging import INFO
from typing import Optional

import numpy as np
import torch
from flwr.common.logger import log


def set_all_random_seeds(seed: Optional[int] = 42) -> None:
    """Set seeds for python random, numpy random, and pytorch random.

    Will no-op if seed is `None`.

    Args:
        seed (int): The seed value to be used for random number generators. Default is 42.
    """
    if seed is None:
        log(INFO, "No seed provided. Using random seed.")
    else:
        log(INFO, f"Setting seed to {seed}")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
