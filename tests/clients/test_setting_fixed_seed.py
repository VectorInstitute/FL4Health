import random

import numpy as np
import pytest
import torch

from fl4health.clients.basic_client import BasicClient
from tests.clients.fixtures import get_apfl_client, get_basic_client  # noqa
from tests.test_utils.models_for_test import ToyConvNet


@pytest.mark.parametrize("model", [ToyConvNet()])
def test_set_fixing_random_seeds(get_basic_client: BasicClient) -> None:  # noqa
    client = get_basic_client
    client._maybe_fix_random_seeds(2023)

    # Check that the random seeds are fixed
    assert pytest.approx(random.random(), abs=0.0001) == 0.3829
    assert pytest.approx(np.random.rand(), abs=0.0001) == 0.3219
    assert pytest.approx(torch.rand(1).item(), abs=0.0001) == 0.4290
    # assert pytest.approx(torch.cuda.FloatTensor(1).uniform_(), abs=0.0001) == 0.4290
