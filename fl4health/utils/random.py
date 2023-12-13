import random
from logging import INFO

import numpy as np
import torch
from flwr.common.logger import log


def set_all_random_seeds(seed: int = 42) -> None:
    """
    set seeds for python random, numpy random, and pytorch random


    Args:
        seed (int): The seed value to be used for random number generators.
    """
    log(INFO, f"Setting seed to {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
