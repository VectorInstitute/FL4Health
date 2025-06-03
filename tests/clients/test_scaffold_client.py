import numpy as np
import pytest
from flwr.common.typing import NDArrays
from opacus.grad_sample.grad_sample_module import GradSampleModule
from opacus.optimizers.optimizer import DPOptimizer
from torch.utils.data import DataLoader

from fl4health.clients.scaffold_client import DPScaffoldClient, ScaffoldClient
from tests.clients.fixtures import get_client  # noqa
from tests.test_utils.models_for_test import Net


@pytest.mark.parametrize("type,model", [(ScaffoldClient, Net())])
def test_compute_parameter_delta(get_client: ScaffoldClient) -> None:  # noqa
    layer_size = 10
    num_layers = 5
    params_1: NDArrays = [np.ones((layer_size)) * 5 for _ in range(num_layers)]
    params_2: NDArrays = [np.zeros((layer_size)) for _ in range(num_layers)]

    client = get_client

    delta_params = client.compute_parameters_delta(params_1, params_2)

    correct_delta_params = [np.ones_like(param_1) * 5 for param_1 in params_1]

    for delta_param, correct_delta_param in zip(delta_params, correct_delta_params):
        assert (delta_param == correct_delta_param).all()


@pytest.mark.parametrize("type,model", [(ScaffoldClient, Net())])
def test_compute_updated_control_variate(get_client: ScaffoldClient) -> None:  # noqa
    layer_size = 10
    num_layers = 5
    local_steps = 5
    delta_model_weights: NDArrays = [np.ones((layer_size)) * 3 for _ in range(num_layers)]
    delta_control_variates: NDArrays = [np.ones((layer_size)) * 100 for _ in range(num_layers)]

    client = get_client

    updated_control_variates = client.compute_updated_control_variates(
        local_steps, delta_model_weights, delta_control_variates
    )
    correct_updated_control_variates = [
        np.ones_like(delta_model_weight) * 160 for delta_model_weight in delta_model_weights
    ]

    for updated_control_variate, correct_updated_control_variate in zip(
        updated_control_variates, correct_updated_control_variates
    ):
        assert (updated_control_variate == correct_updated_control_variate).all()


@pytest.mark.parametrize("type,model", [(DPScaffoldClient, Net())])
def test_dp_scaffold_client(get_client: DPScaffoldClient) -> None:  # noqa
    client: DPScaffoldClient = get_client
    client.setup_opacus_objects({})

    assert isinstance(client.model, GradSampleModule)
    assert isinstance(client.optimizers["global"], DPOptimizer)
    assert isinstance(client.train_loader, DataLoader)

    assert hasattr(client, "client_control_variates")
    assert hasattr(client, "server_control_variates")
    assert hasattr(client, "client_control_variates_updates")
