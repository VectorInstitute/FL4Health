import random

import numpy as np
from flwr.common import (
    Code,
    EvaluateRes,
    FitRes,
    Metrics,
    NDArrays,
    Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import SimpleClientManager
from flwr.server.client_proxy import ClientProxy

from fl4health.client_managers.poisson_sampling_manager import PoissonSamplingClientManager
from fl4health.strategies.model_merge_strategy import ModelMergeStrategy
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
clients_res: list[tuple[ClientProxy, FitRes]] = [
    (CustomClientProxy("c0"), client0_res),
    (CustomClientProxy("c1"), client1_res),
    (CustomClientProxy("c2"), client2_res),
    (CustomClientProxy("c3"), client3_res),
]


def metrics_aggregation(to_aggregate: list[tuple[int, Metrics]]) -> Metrics:
    # Select last set of metrics (dummy for test)
    return to_aggregate[-1][1]


evaluate_metrics_aggregation_fn = metrics_aggregation
fit_metrics_aggregation_fn = metrics_aggregation

weighted_strategy = ModelMergeStrategy(
    evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
    fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
    fraction_evaluate=0.8,
    fraction_fit=0.8,
    weighted_aggregation=True,
)
unweighted_strategy = ModelMergeStrategy(
    evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
    fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
    weighted_aggregation=False,
)


def test_aggregate_fit() -> None:
    empty_results, empty_metrics = weighted_strategy.aggregate_fit(server_round=1, results=[], failures=[])
    assert empty_results is None and not empty_metrics

    parameters, metrics = weighted_strategy.aggregate_fit(server_round=1, results=clients_res, failures=[])
    assert metrics["metric"] == 0.4

    # First layer weighted aggregate should be all 3.0
    weighted_target_1 = np.ones((3, 3)) * (3.0)
    # Second layer weighted aggregate should be all 25/8
    weighted_target_2 = np.ones((4, 4)) * (25.0 / 8.0)

    assert parameters is not None
    aggregated_ndarrays = parameters_to_ndarrays(parameters)
    assert np.allclose(weighted_target_1, aggregated_ndarrays[0])
    assert np.allclose(weighted_target_2, aggregated_ndarrays[1])

    parameters, metrics = unweighted_strategy.aggregate_fit(server_round=1, results=clients_res, failures=[])
    assert metrics["metric"] == 0.4

    # First layer unweighted aggregate should be all 9/4
    unweighted_target_1 = np.ones((3, 3)) * (9.0 / 4.0)
    # Second layer unweighted aggregate should be all 5/2
    unweighted_target_2 = np.ones((4, 4)) * (5.0 / 2.0)

    assert parameters is not None
    aggregated_ndarrays = parameters_to_ndarrays(parameters)
    assert np.allclose(unweighted_target_1, aggregated_ndarrays[0])
    assert np.allclose(unweighted_target_2, aggregated_ndarrays[1])


def construct_evaluate_res(loss: float, metric: float, num_examples: int) -> EvaluateRes:
    return EvaluateRes(status=Status(Code.OK, ""), num_examples=num_examples, loss=loss, metrics={"metric": metric})


client0_eval_res = construct_evaluate_res(1.0, 0.1, 50)
client1_eval_res = construct_evaluate_res(1.0, 0.2, 50)
client2_eval_res = construct_evaluate_res(3.0, 0.3, 100)
client3_eval_res = construct_evaluate_res(4.0, 0.4, 200)
clients_eval_res: list[tuple[ClientProxy, EvaluateRes]] = [
    (CustomClientProxy("c0"), client0_eval_res),
    (CustomClientProxy("c1"), client1_eval_res),
    (CustomClientProxy("c2"), client2_eval_res),
    (CustomClientProxy("c3"), client3_eval_res),
]


def test_aggregate_evaluate() -> None:
    empty_agg_loss, empty_metrics = weighted_strategy.aggregate_evaluate(server_round=1, results=[], failures=[])
    assert empty_agg_loss is None and not empty_metrics

    agg_loss, metrics = weighted_strategy.aggregate_evaluate(server_round=1, results=clients_eval_res, failures=[])
    assert metrics["metric"] == 0.4 and agg_loss is None

    agg_loss, metrics = unweighted_strategy.aggregate_evaluate(server_round=1, results=clients_eval_res, failures=[])
    assert metrics["metric"] == 0.4 and agg_loss is None


client_proxies = [client_proxy for client_proxy, _ in clients_res]
client_params = [fit_res.parameters for _, fit_res in clients_res]
poisson_client_manager = PoissonSamplingClientManager()
poisson_client_manager.clients = {proxy.cid: proxy for proxy in client_proxies}
simple_client_manager = SimpleClientManager()
simple_client_manager.clients = {proxy.cid: proxy for proxy in client_proxies}


def test_configure_fit() -> None:
    np.random.seed(42)
    random.seed(42)

    # Just need one set of parameters, sample using poisson client manager
    poisson_config_res = weighted_strategy.configure_fit(
        server_round=1, parameters=client_params[0], client_manager=poisson_client_manager
    )
    assert len(poisson_config_res) == 3
    # We should skip the second client in this sample
    assert poisson_config_res[1][0].cid == "c2"

    # Just need one set of parameters, sample using simple client manager
    simple_config_res = weighted_strategy.configure_fit(
        server_round=1, parameters=client_params[0], client_manager=simple_client_manager
    )
    assert len(simple_config_res) == 3
    # Client three should be the second chosen.
    assert simple_config_res[1][0].cid == "c3"


def test_configure_eval() -> None:
    np.random.seed(42)
    random.seed(42)

    # Just need one set of parameters, sample using poisson client manager
    poisson_config_res = weighted_strategy.configure_evaluate(
        server_round=1, parameters=client_params[0], client_manager=poisson_client_manager
    )
    assert len(poisson_config_res) == 3
    # We should skip the second client in this sample
    assert poisson_config_res[1][0].cid == "c2"

    # Just need one set of parameters, sample using simple client manager
    simple_config_res = weighted_strategy.configure_evaluate(
        server_round=1, parameters=client_params[0], client_manager=simple_client_manager
    )
    assert len(simple_config_res) == 3
    # Client three should be the second chosen.
    assert simple_config_res[1][0].cid == "c3"
