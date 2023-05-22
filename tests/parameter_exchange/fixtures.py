from typing import List

import numpy as np
import pytest
from flwr.common.typing import NDArrays


@pytest.fixture
def get_ndarrays(layer_sizes: List[List[int]]) -> NDArrays:
    ndarrays = [np.ones(tuple(size)) for size in layer_sizes]
    return ndarrays
