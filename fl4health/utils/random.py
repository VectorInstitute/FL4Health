import random
import uuid
from logging import INFO
from typing import Any

import numpy as np
import torch
from flwr.common.logger import log


def set_all_random_seeds(
    seed: int | None = 42, use_deterministic_torch_algos: bool = False, disable_torch_benchmarking: bool = False
) -> None:
    """
    Set seeds for python random, numpy random, and pytorch random. It also offers the option to force pytorch to use
    deterministic algorithm for certain methods and layers.

    See:

    https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html)

    for more details. Finally, it allows one to disable cuda benchmarking, which can also affect the determinism of
    pytorch training outside of random seeding. For more information on reproducibility in pytorch see:

    https://pytorch.org/docs/stable/notes/randomness.html

    **NOTE**: If the ``use_deterministic_torch_algos`` flag is set to True, you may need to set the environment
    variable ``CUBLAS_WORKSPACE_CONFIG`` to something like ``:4096:8``, to avoid CUDA errors. Additional documentation
    may be found here:

    https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility

    Args:
        seed (int | None, optional): The seed value to be used for random number generators. Default is 42. Seed
            setting will no-op if the seed is explicitly set to None.
        use_deterministic_torch_algos (bool, optional): Whether or not to set ``torch.use_deterministic_algorithms`` to
            True. Defaults to False.
        disable_torch_benchmarking (bool, optional): Whether to explicitly disable cuda benchmarking in torch
            processes. Defaults to False.
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
    the effects of ``set_all_random_seeds``.
    """
    log(INFO, "Setting all random seeds to None. Reverting torch determinism settings")
    random.seed(None)
    np.random.seed(None)
    torch.seed()
    torch.use_deterministic_algorithms(False)


def save_random_state() -> tuple[tuple[Any, ...], dict[str, Any], torch.Tensor]:
    """
    Save the state of the random number generators for Python, NumPy, and PyTorch. This will allow you to restore the
    state of the random number generators at a later time.

    Returns:
        (tuple[tuple[Any, ...], dict[str, Any], torch.Tensor]): A tuple containing the state of the random number
        generators for Python, NumPy, and PyTorch.
    """
    log(INFO, "Saving random state.")
    random_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    return random_state, numpy_state, torch_state


def restore_random_state(
    random_state: tuple[Any, ...], numpy_state: dict[str, Any], torch_state: torch.Tensor
) -> None:
    """
    Restore the state of the random number generators for Python, NumPy, and PyTorch. This will allow you to restore
    the state of the random number generators to a previously saved state.

    Args:
        random_state (tuple[Any, ...]): The state of the Python random number generator.
        numpy_state (dict[str, Any]): The state of the NumPy random number generator.
        torch_state (torch.Tensor): The state of the PyTorch random number generator.
    """
    log(INFO, "Restoring random state.")
    random.setstate(random_state)
    np.random.set_state(numpy_state)
    torch.set_rng_state(torch_state)


def generate_hash(length: int = 8) -> str:
    """
    Generates unique hash used as id for client.

    **NOTE**: This generation is unaffected by setting of random seeds.

    Args:
       length (int): The length of the hash generated. Maximum length is 32.

    Returns:
        (str): hash
    """
    return str(uuid.uuid4()).replace("-", "")[:length]
