from pathlib import Path

import numpy as np
import pytest
import torch
from flwr.common import Config

from fl4health.clients.clipping_client import NumpyClippingClient
from fl4health.metrics import Accuracy


class DummyClippingClient(NumpyClippingClient):
    def setup_client(self, config: Config) -> None:
        raise NotImplementedError


def test_weight_update_and_clipping() -> None:
    clipping_client = DummyClippingClient(Path(""), [Accuracy("accuracy")], torch.device("cpu"))
    clipping_client.adaptive_clipping = True
    clipping_client.clipping_bound = 1.0
    n_layers = 4
    clipping_client.initial_weights = [2.0 * np.ones((2, 3, 3)) for _ in range(n_layers)]
    new_weights = [4.0 * np.ones((2, 3, 3)) for _ in range(n_layers)]
    clipped_weight_update, clipping_bit = clipping_client.compute_weight_update_and_clip(new_weights)

    assert clipping_bit == 0.0
    layer_0_clipped_weight_update = clipped_weight_update[0]
    assert pytest.approx(layer_0_clipped_weight_update[0, 0, 0], abs=0.0001) == 0.11785


def test_clipping_bit_flip() -> None:
    clipping_client = DummyClippingClient(Path(""), [Accuracy("accuracy")], torch.device("cpu"))
    clipping_client.adaptive_clipping = True
    clipping_client.clipping_bound = 9.0
    n_layers = 4
    clipping_client.initial_weights = [2.0 * np.ones((2, 3, 3)) for _ in range(n_layers)]
    new_weights = [3.0 * np.ones((2, 3, 3)) for _ in range(n_layers)]
    clipped_weight_update, clipping_bit = clipping_client.compute_weight_update_and_clip(new_weights)

    assert clipping_bit == 1.0
    layer_0_clipped_weight_update = clipped_weight_update[0]
    assert pytest.approx(layer_0_clipped_weight_update[0, 0, 0], abs=0.0001) == 1.0
