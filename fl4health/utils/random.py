import random
import uuid
from logging import INFO
from typing import Optional

import numpy as np
import torch
from flwr.common.logger import log


def set_all_random_seeds(
    seed: Optional[int] = 42, use_deterministic_torch_algos: bool = False, disable_torch_benchmarking: bool = False
) -> None:
    """
    Set seeds for python random, numpy random, and pytorch random. It also offers the option to force pytorch to use
    deterministic algorithm for certain methods and layers see:
    https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html) for more details. Finally, it
    allows one to disable cuda benchmarking, which can also affect the determinism of pytorch training outside of
    random seeding. For more information on reproducibility in pytorch see:
    https://pytorch.org/docs/stable/notes/randomness.html

    NOTE: If the use_deterministic_torch_algos flag is set to True, you may need to set the environment variable
    CUBLAS_WORKSPACE_CONFIG to something like :4096:8, to avoid CUDA errors. Additional documentation may be found
    here: https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility

    Args:
        seed (Optional[int], optional): The seed value to be used for random number generators. Default is 42. Seed
            setting will no-op if the seed is explicitly set to None
        use_deterministic_torch_algos (bool, optional): Whether or not to set torch.use_deterministic_algorithms to
            True. Defaults to False.
        disable_torch_benchmarking (bool, optional): Whether to explicitly disable cuda benchmarking in
            torch processes. Defaults to False.
    """
    if seed is None:
        log(INFO, "No seed provided. Using random seed.")
    else:
        log(INFO, f"Setting random seeds to {seed}.")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    if use_deterministic_torch_algos:
        log(INFO, "Setting torch.use_deterministic_algorithms to True.")
        # warn_only is set to true so that layers and components without deterministic algorithms available will
        # warn the user that they don't exist, but won't take down the process with an exception.
        torch.use_deterministic_algorithms(True, warn_only=True)
    if disable_torch_benchmarking:
        log(INFO, "Disabling CUDA algorithm benchmarking.")
        torch.backends.cudnn.benchmark = False


def unset_all_random_seeds() -> None:
    """
    Set random seeds for Python random, NumPy, and PyTorch to None. Running this function would undo
    the effects of set_all_random_seeds.
    """
    log(INFO, "Setting all random seeds to None. Reverting torch determinism settings")
    random.seed(None)
    np.random.seed(None)
    torch.seed()
    torch.use_deterministic_algorithms(False)


def generate_hash(length: int = 8) -> str:
    """
    Generates unique hash used as id for client.
    NOTE: This generation is unaffected by setting of random seeds.

    Args:
       length (int): The length of the hash generated. Maximum length is 32

    Returns:
        str: hash
    """
    return str(uuid.uuid4()).replace("-", "")[:length]
