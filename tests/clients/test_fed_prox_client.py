from pathlib import Path

import pytest
import torch
from flwr.common import Config

from fl4health.clients.fed_prox_client import FedProxClient
from fl4health.parameter_exchange.full_exchanger import FullParameterExchanger
from fl4health.utils.metrics import Accuracy
from tests.clients.small_model import TestCNN


class DummyFedProxClient(FedProxClient):
    def setup_client(self, _: Config) -> None:
        self.model = TestCNN()
        self.parameter_exchanger = FullParameterExchanger()


def test_setting_initial_weights() -> None:
    torch.manual_seed(42)
    fed_prox_client = DummyFedProxClient(Path(""), [Accuracy()], torch.device("cpu"))
    config: Config = {}

    fed_prox_client.setup_client(config)
    params = [val.cpu().numpy() for _, val in fed_prox_client.model.state_dict().items()]
    fed_prox_client.set_parameters(params, config)

    assert fed_prox_client.initial_tensor is not None
    # Tensors should be conv1 weights, biases, conv2 weights, biases, fc1 weights, biases (so 6 total)
    assert len(fed_prox_client.initial_tensor) == 6
    # Make sure that each layer tensor has a non-zero norm
    for layer_tensor in fed_prox_client.initial_tensor:
        assert torch.norm(layer_tensor) > 0.0


def test_forming_proximal_loss() -> None:
    torch.manual_seed(42)
    fed_prox_client = DummyFedProxClient(Path(""), [Accuracy()], torch.device("cpu"))
    config: Config = {}

    fed_prox_client.setup_client(config)
    params = [val.cpu().numpy() for _, val in fed_prox_client.model.state_dict().items()]
    fed_prox_client.set_parameters(params, config)

    # We've taken no training steps so the proximal loss should be 0.0
    assert fed_prox_client.get_proximal_loss().detach().item() == 0.0

    perturbed_params = [layer_weights + 0.1 for layer_weights in params]
    # Circumventing the set_parameters function to update the model weights
    fed_prox_client.parameter_exchanger.pull_parameters(perturbed_params, fed_prox_client.model, config)

    proximal_loss = fed_prox_client.get_proximal_loss()

    assert pytest.approx(proximal_loss.detach().item(), abs=0.002) == (1.5 + 0.06 + 24.0 + 81.92 + 0.16 + 0.32)
