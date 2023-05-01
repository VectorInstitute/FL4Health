from pathlib import Path
from typing import List

import numpy as np
import torch
from flwr.common import NDArrays, Parameters, ndarrays_to_parameters

from fl4health.clients.scaffold_client import ScaffoldClient
from fl4health.strategies.scaffold import Scaffold
from fl4health.utils.metrics import Accuracy, Metric


def test_aggregate() -> None:
    layer_size = 10
    num_layers = 5
    num_clients = 3
    ndarrays: NDArrays = [np.ones((layer_size)) for _ in range(num_layers)]
    params: Parameters = ndarrays_to_parameters(ndarrays)
    strat = Scaffold(initial_parameters=params)

    client_ndarrays = [[ndarray * (client_num + 1) for ndarray in ndarrays] for client_num in range(num_clients)]
    new_ndarrays = strat.aggregate(client_ndarrays)

    correct_ndarrays = [[ndarray * 2 for ndarray in ndarrays] for _ in range(num_clients)]

    for new_ndarray, correct_ndarray in zip(new_ndarrays, correct_ndarrays):
        assert (new_ndarray == correct_ndarray).all()


def test_compute_parameter_delta() -> None:

    layer_size = 10
    num_layers = 5
    params_1: NDArrays = [np.ones((layer_size)) * 5 for _ in range(num_layers)]
    params_2: NDArrays = [np.zeros((layer_size)) for _ in range(num_layers)]

    path = Path("./")
    device = torch.device("cpu")
    accuracy_metric = Accuracy()
    metrics: List[Metric] = [accuracy_metric]
    client = ScaffoldClient(path, metrics, device)

    delta_params = client.compute_parameter_delta(params_1, params_2)

    correct_delta_params = [np.ones_like(param_1) * 5 for param_1 in params_1]

    for delta_param, correct_delta_param in zip(delta_params, correct_delta_params):
        assert (delta_param == correct_delta_param).all()


def test_compute_updated_control_variate() -> None:

    layer_size = 10
    num_layers = 5
    local_steps = 5
    delta_model_weights: NDArrays = [np.ones((layer_size)) * 3 for _ in range(num_layers)]
    delta_control_variates: NDArrays = [np.ones((layer_size)) * 100 for _ in range(num_layers)]

    path = Path("./")
    device = torch.device("cpu")
    accuracy_metric = Accuracy()
    metrics: List[Metric] = [accuracy_metric]
    client = ScaffoldClient(path, metrics, device)
    client.learning_rate_local = 0.01

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
