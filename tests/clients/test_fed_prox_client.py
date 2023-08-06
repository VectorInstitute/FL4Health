import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr.common import Config

from fl4health.clients.fed_prox_client import FedProxClient
from tests.clients.fixtures import get_client  # noqa


class SmallCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        return x


class LinearTransform(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 3, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


@pytest.mark.parametrize("type,model", [(FedProxClient, SmallCNN())])
def test_setting_initial_weights(get_client: FedProxClient) -> None:  # noqa
    torch.manual_seed(42)
    fed_prox_client = get_client
    config: Config = {}

    fed_prox_client.setup_client(config)
    params = [val.cpu().numpy() for _, val in fed_prox_client.model.state_dict().items()]
    proximal_weight = 0.0
    additional_variables = {}
    additional_variables["proximal_weight"] = [np.array(proximal_weight)]
    packed_params = params + [np.array(list(additional_variables.keys()))]
    for _, values in additional_variables.items():
        packed_params += values

    # Circumventing the set_parameters function to update the model weights
    fed_prox_client.set_parameters(packed_params, config)

    assert fed_prox_client.initial_tensors is not None
    # Tensors should be conv1 weights, biases, conv2 weights, biases, fc1 weights, biases (so 6 total)
    assert len(fed_prox_client.initial_tensors) == 6
    # Make sure that each layer tensor has a non-zero norm
    for layer_tensor in fed_prox_client.initial_tensors:
        assert torch.linalg.norm(layer_tensor) > 0.0


@pytest.mark.parametrize("type,model", [(FedProxClient, SmallCNN())])
def test_forming_proximal_loss(get_client: FedProxClient) -> None:  # noqa
    torch.manual_seed(42)
    fed_prox_client = get_client
    config: Config = {}

    fed_prox_client.setup_client(config)
    params = [val.cpu().numpy() for _, val in fed_prox_client.model.state_dict().items()]
    proximal_weight = 0.0
    additional_variables = {}
    additional_variables["proximal_weight"] = [np.array(proximal_weight)]
    packed_params = params + [np.array(list(additional_variables.keys()))]
    for _, values in additional_variables.items():
        packed_params += values
    fed_prox_client.set_parameters(packed_params, config)

    # We've taken no training steps so the proximal loss should be 0.0
    assert fed_prox_client.get_proximal_loss().detach().item() == 0.0

    perturbed_params = [layer_weights + 0.1 for layer_weights in params]
    perturbed_proximal_weight = 0.0
    additional_variables = {}
    additional_variables["proximal_weight"] = [np.array(perturbed_proximal_weight)]
    packed_perturbed_params = perturbed_params + [np.array(list(additional_variables.keys()))]
    for _, values in additional_variables.items():
        packed_perturbed_params += values

    # Circumventing the set_parameters function to update the model weights
    fed_prox_client.set_parameters(packed_perturbed_params, config)

    proximal_loss = fed_prox_client.get_proximal_loss()

    assert pytest.approx(proximal_loss.detach().item(), abs=0.002) == (fed_prox_client.proximal_weight / 2.0) * (
        1.5 + 0.06 + 24.0 + 81.92 + 0.16 + 0.32
    )


@pytest.mark.parametrize("type,model", [(FedProxClient, LinearTransform())])
def test_proximal_loss_derivative(get_client: FedProxClient) -> None:  # noqa
    torch.manual_seed(42)
    fed_prox_client = get_client
    config: Config = {}

    fed_prox_client.setup_client(config)
    params = [val.cpu().numpy() for _, val in fed_prox_client.model.state_dict().items()]
    proximal_weight = 0.0
    additional_variables = {}
    additional_variables["proximal_weight"] = [np.array(proximal_weight)]
    packed_params = params + [np.array(list(additional_variables.keys()))]
    for _, values in additional_variables.items():
        packed_params += values
    fed_prox_client.set_parameters(packed_params, config)

    perturbed_params = [layer_weights + 0.1 for layer_weights in params]
    perturbed_proximal_weight = 1.0
    additional_variables = {}
    additional_variables["proximal_weight"] = [np.array(perturbed_proximal_weight)]
    packed_perturbed_params = perturbed_params + [np.array(list(additional_variables.keys()))]
    for _, values in additional_variables.items():
        packed_perturbed_params += values

    # Circumventing the set_parameters function to update the model weights
    fed_prox_client.set_parameters(packed_perturbed_params, config)

    proximal_loss = fed_prox_client.get_proximal_loss()
    proximal_loss.backward()
    linear_gradient = list(fed_prox_client.model.parameters())[0].grad

    # We have a linear layer and we have perturbed the weights by 0.1. The proximal loss is
    # \mu/2 \cdot || w_t - w ||_2^2
    # So the derivative is -\mu (w_t - w). Since \mu = 0.1 and we perturb the weights by 0.1, each component of the
    # linear layer has derivative -0.1(-0.1) = 0.01
    torch.testing.assert_close(
        linear_gradient, torch.tensor([[0.01, 0.01], [0.01, 0.01], [0.01, 0.01]]), atol=0.0001, rtol=1.0
    )


@pytest.mark.parametrize("type,model", [(FedProxClient, SmallCNN())])
def test_setting_proximal_weight(get_client: FedProxClient) -> None:  # noqa
    torch.manual_seed(42)
    fed_prox_client = get_client
    config: Config = {}

    fed_prox_client.setup_client(config)
    params = [val.cpu().numpy() for _, val in fed_prox_client.model.state_dict().items()]
    proximal_weight = 0.0
    additional_variables = {}
    additional_variables["proximal_weight"] = [np.array(proximal_weight)]
    packed_params = params + [np.array(list(additional_variables.keys()))]
    for _, values in additional_variables.items():
        packed_params += values
    fed_prox_client.set_parameters(packed_params, config)

    # We've taken no training steps so the proximal loss should be 0.0
    assert fed_prox_client.get_proximal_loss().detach().item() == 0.0

    perturbed_params = [layer_weights + 0.1 for layer_weights in params]
    perturbed_proximal_weight = 1.0
    additional_variables = {}
    additional_variables["proximal_weight"] = [np.array(perturbed_proximal_weight)]
    packed_perturbed_params = perturbed_params + [np.array(list(additional_variables.keys()))]
    for _, values in additional_variables.items():
        packed_perturbed_params += values

    # Circumventing the set_parameters function to update the model weights
    fed_prox_client.set_parameters(packed_perturbed_params, config)

    assert fed_prox_client.proximal_weight == perturbed_proximal_weight
