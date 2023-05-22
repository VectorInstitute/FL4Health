import pytest
import torch
from flwr.common import Config

from fl4health.clients.fed_prox_client import FedProxClient
from tests.clients.fixtures import get_client  # noqa
from tests.clients.small_models import LinearTransform, TestCNN


@pytest.mark.parametrize("type,model", [(FedProxClient, TestCNN())])
def test_setting_initial_weights(get_client: FedProxClient) -> None:  # noqa
    torch.manual_seed(42)
    fed_prox_client = get_client  # noqa
    config: Config = {}

    fed_prox_client.setup_client(config)
    params = [val.cpu().numpy() for _, val in fed_prox_client.model.state_dict().items()]
    fed_prox_client.set_parameters(params, config)

    assert fed_prox_client.initial_tensors is not None
    # Tensors should be conv1 weights, biases, conv2 weights, biases, fc1 weights, biases (so 6 total)
    assert len(fed_prox_client.initial_tensors) == 6
    # Make sure that each layer tensor has a non-zero norm
    for layer_tensor in fed_prox_client.initial_tensors:
        assert torch.linalg.norm(layer_tensor) > 0.0


@pytest.mark.parametrize("type,model", [(FedProxClient, TestCNN())])
def test_forming_proximal_loss(get_client: FedProxClient) -> None:  # noqa
    torch.manual_seed(42)
    fed_prox_client = get_client  # noqa
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

    assert pytest.approx(proximal_loss.detach().item(), abs=0.002) == (fed_prox_client.proximal_weight / 2.0) * (
        1.5 + 0.06 + 24.0 + 81.92 + 0.16 + 0.32
    )


@pytest.mark.parametrize("type,model", [(FedProxClient, LinearTransform())])
def test_proximal_loss_derivative(get_client: FedProxClient) -> None:  # noqa
    torch.manual_seed(42)
    fed_prox_client = get_client  # noqa
    config: Config = {}

    fed_prox_client.setup_client(config)
    params = [val.cpu().numpy() for _, val in fed_prox_client.model.state_dict().items()]
    fed_prox_client.set_parameters(params, config)

    perturbed_params = [layer_weights + 0.1 for layer_weights in params]
    # Circumventing the set_parameters function to update the model weights
    fed_prox_client.parameter_exchanger.pull_parameters(perturbed_params, fed_prox_client.model, config)

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
