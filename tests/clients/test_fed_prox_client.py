from collections import OrderedDict

import pytest
import torch
from flwr.common import Config

from fl4health.clients.fed_prox_client import FedProxClient
from tests.clients.fixtures import get_client  # noqa
from tests.test_utils.models_for_test import LinearTransform, SmallCnn


@pytest.mark.parametrize("type,model", [(FedProxClient, SmallCnn())])
def test_setting_initial_weights(get_client: FedProxClient) -> None:  # noqa
    torch.manual_seed(42)
    fed_prox_client = get_client
    config: Config = {}

    params = [val.cpu().numpy() for _, val in fed_prox_client.model.state_dict().items()]
    proximal_weight = 0.0
    packed_params = fed_prox_client.parameter_exchanger.pack_parameters(params, proximal_weight)

    fed_prox_client.set_parameters(packed_params, config)

    assert fed_prox_client.initial_tensors is not None
    # Tensors should be conv1 weights, biases, conv2 weights, biases, fc1 weights, biases (so 6 total)
    assert len(fed_prox_client.initial_tensors) == 6
    # Make sure that each layer tensor has a non-zero norm
    for layer_tensor in fed_prox_client.initial_tensors:
        assert torch.linalg.norm(layer_tensor) > 0.0


@pytest.mark.parametrize("type,model", [(FedProxClient, SmallCnn())])
def test_forming_proximal_loss(get_client: FedProxClient) -> None:  # noqa
    torch.manual_seed(42)
    fed_prox_client = get_client
    config: Config = {}

    params = [val.cpu().numpy() for _, val in fed_prox_client.model.state_dict().items()]
    proximal_weight = 0.0
    packed_params = fed_prox_client.parameter_exchanger.pack_parameters(params, proximal_weight)
    fed_prox_client.set_parameters(packed_params, config)

    # We've taken no training steps so the proximal loss should be 0.0
    assert fed_prox_client.get_proximal_loss().detach().item() == 0.0

    perturbed_params = [layer_weights + 0.1 for layer_weights in params]
    perturbed_proximal_weight = 0.0
    packed_perturbed_params = fed_prox_client.parameter_exchanger.pack_parameters(
        perturbed_params, perturbed_proximal_weight
    )

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

    params = [val.cpu().numpy() for _, val in fed_prox_client.model.state_dict().items()]
    proximal_weight = 0.0
    packed_params = fed_prox_client.parameter_exchanger.pack_parameters(params, proximal_weight)
    fed_prox_client.set_parameters(packed_params, config)

    perturbed_params = [layer_weights + 0.1 for layer_weights in params]
    perturbed_proximal_weight = 1.0
    packed_perturbed_params = fed_prox_client.parameter_exchanger.pack_parameters(
        perturbed_params, perturbed_proximal_weight
    )

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


@pytest.mark.parametrize("type,model", [(FedProxClient, SmallCnn())])
def test_setting_proximal_weight(get_client: FedProxClient) -> None:  # noqa
    torch.manual_seed(42)
    fed_prox_client = get_client
    config: Config = {}

    params = [val.cpu().numpy() for _, val in fed_prox_client.model.state_dict().items()]
    proximal_weight = 0.0
    packed_params = fed_prox_client.parameter_exchanger.pack_parameters(params, proximal_weight)
    fed_prox_client.set_parameters(packed_params, config)

    # We've taken no training steps so the proximal loss should be 0.0
    assert fed_prox_client.get_proximal_loss().detach().item() == 0.0

    perturbed_params = [layer_weights + 0.1 for layer_weights in params]
    perturbed_proximal_weight = 1.0
    packed_perturbed_params = fed_prox_client.parameter_exchanger.pack_parameters(
        perturbed_params, perturbed_proximal_weight
    )

    fed_prox_client.set_parameters(packed_perturbed_params, config)

    assert fed_prox_client.proximal_weight == perturbed_proximal_weight


@pytest.mark.parametrize("type,model", [(FedProxClient, SmallCnn())])
def test_compute_loss(get_client: FedProxClient) -> None:  # noqa
    torch.manual_seed(42)
    fed_prox_client = get_client
    config: Config = {}
    fed_prox_client.criterion = torch.nn.CrossEntropyLoss()

    params = [val.cpu().numpy() for _, val in fed_prox_client.model.state_dict().items()]
    proximal_weight = 1.0
    packed_params = fed_prox_client.parameter_exchanger.pack_parameters(params, proximal_weight)
    fed_prox_client.set_parameters(packed_params, config)

    # We've taken no training steps so the proximal loss should be 0.0
    assert fed_prox_client.get_proximal_loss().detach().item() == 0.0

    perturbed_params = [layer_weights + 0.1 for layer_weights in params]

    params_dict = zip(fed_prox_client.model.state_dict().keys(), perturbed_params)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    fed_prox_client.model.load_state_dict(state_dict, strict=True)

    preds = {"prediction": torch.tensor([[1.0, 0.0], [0.0, 1.0]])}
    target = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    loss = fed_prox_client.compute_loss(preds, {}, target)
    assert isinstance(loss.backward, torch.Tensor)
    assert pytest.approx(0.8132616, abs=0.0001) == loss.checkpoint.item()
    assert loss.checkpoint.item() != loss.backward.item()
