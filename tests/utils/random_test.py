import random

import numpy as np
import pytest
import torch

from fl4health.utils.random import set_all_random_seeds


def test_set_fixing_random_seeds() -> None:  # noqa
    # Set the random seeds
    set_all_random_seeds(2023)

    # Check that the random seeds are fixed
    assert pytest.approx(random.random(), abs=0.0001) == 0.3829
    assert pytest.approx(np.random.rand(), abs=0.0001) == 0.3219
    assert pytest.approx(torch.rand(1).item(), abs=0.0001) == 0.4290
