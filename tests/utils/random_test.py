import random

import numpy as np
import pytest
import torch

from fl4health.utils.random import restore_random_state, save_random_state, set_all_random_seeds


def test_set_fixing_random_seeds() -> None:
    # Set the random seeds
    set_all_random_seeds(2023)

    # Check that the random seeds are fixed
    assert pytest.approx(random.random(), abs=0.0001) == 0.3829
    assert pytest.approx(np.random.rand(), abs=0.0001) == 0.3219
    assert pytest.approx(torch.rand(1).item(), abs=0.0001) == 0.4290


def test_saving_and_restoring_random_state() -> None:
    # First set the random seeds
    set_all_random_seeds(2025)
    # Perform some generation
    random.random()
    np.random.rand()
    torch.rand(1)

    # Save the state
    random_state, np_random_state, torch_random_state = save_random_state()

    # Do a touch more
    rand_float = random.random()
    np_rand_float = np.random.rand()
    torch_rand_float = torch.rand(1)

    # Restore random state
    restore_random_state(random_state, np_random_state, torch_random_state)

    # Make sure the next random values match
    assert pytest.approx(rand_float, abs=1e-6) == random.random()
    assert pytest.approx(np_rand_float, abs=1e-6) == np.random.rand()
    assert pytest.approx(torch_rand_float, abs=1e-6) == torch.rand(1)


def test_setting_cuda_determinism_and_benchmarks() -> None:
    # Set the random seeds and determinism settings (only actually have an effect if using cuda)
    set_all_random_seeds(2023, use_deterministic_torch_algos=True, disable_torch_benchmarking=True)
