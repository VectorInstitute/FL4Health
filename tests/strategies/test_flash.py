import random
from typing import List, Tuple

import numpy as np
from flwr.common import Code, FitRes, Metrics, NDArrays, Status, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy

from fl4health.strategies.flash import FLASH
from tests.test_utils.custom_client_proxy import CustomClientProxy


def construct_fit_res(parameters: NDArrays, metric: float, num_examples: int) -> FitRes:
    return FitRes(
        status=Status(Code.OK, ""),
        parameters=ndarrays_to_parameters(parameters),
        num_examples=num_examples,
        metrics={"metric": metric},
    )


client0_res = construct_fit_res([np.ones((3, 3)), np.ones((4, 4))], 0.1, 50)
client1_res = construct_fit_res([np.ones((3, 3)), np.full((4, 4), 2)], 0.2, 50)
client2_res = construct_fit_res([np.full((3, 3), 3), np.full((4, 4), 3)], 0.3, 100)
client3_res = construct_fit_res([np.full((3, 3), 4), np.full((4, 4), 4)], 0.4, 200)
clients_res: List[Tuple[ClientProxy, FitRes]] = [
    (CustomClientProxy("c0"), client0_res),
    (CustomClientProxy("c1"), client1_res),
    (CustomClientProxy("c2"), client2_res),
    (CustomClientProxy("c3"), client3_res),
]


def metrics_aggregation(to_aggregate: List[Tuple[int, Metrics]]) -> Metrics:
    # Select last set of metrics (dummy for test)
    return to_aggregate[-1][1]


evaluate_metrics_aggregation_fn = metrics_aggregation
fit_metrics_aggregation_fn = metrics_aggregation

flash_strategy = FLASH(
    evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
    fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
    fraction_evaluate=0.8,
    fraction_fit=0.8,
    min_fit_clients=2,
    min_evaluate_clients=2,
    min_available_clients=2,
    accept_failures=True,
    initial_parameters=ndarrays_to_parameters([np.zeros((3, 3)), np.zeros((4, 4))]),
    eta=0.1,
    eta_l=0.1,
    beta_1=0.9,
    beta_2=0.99,
    tau=1e-9,
)


def test_flash_aggregate_fit() -> None:
    empty_results, empty_metrics = flash_strategy.aggregate_fit(server_round=1, results=[], failures=[])
    assert empty_results is None and not empty_metrics

    parameters, metrics = flash_strategy.aggregate_fit(server_round=1, results=clients_res, failures=[])
    assert metrics["metric"] == 0.4

    # First layer weighted aggregate should be all -0.00352075
    weighted_target_1 = np.ones((3, 3)) * (-0.00352075)
    # Second layer weighted aggregate should be all -0.00337517
    weighted_target_2 = np.ones((4, 4)) * (-0.00337517)

    assert parameters is not None
    aggregated_ndarrays = parameters_to_ndarrays(parameters)
    assert np.allclose(weighted_target_1, aggregated_ndarrays[0])
    assert np.allclose(weighted_target_2, aggregated_ndarrays[1])
