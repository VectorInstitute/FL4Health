from fl4health.strategies.basic_fedavg import BasicFedAvg

from logging import DEBUG, INFO, WARNING
from typing import Callable, Dict, List, Optional, Tuple, Union

from flwr.common import (
    FitRes,
    GetPropertiesIns,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from fl4health.client_managers.base_sampling_manager import BaseFractionSamplingManager
from fl4health.strategies.aggregate_utils import aggregate_results
from fl4health.strategies.basic_fedavg import BasicFedAvg

Requests = List[Tuple[ClientProxy, GetPropertiesIns]]
Request = Tuple[ClientProxy, GetPropertiesIns]

from functools import reduce

import numpy as np


class CentralDPStrategy(BasicFedAvg):
    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        weighted_aggregation: bool = True,
        weighted_eval_losses: bool = True,
    ) -> None:
        # NOTE currently secure aggregation supports no dropouts, hence fraction_fit_fit = 1
        # I will define on_fit_config_fn to avoid complication
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
            weighted_aggregation=weighted_aggregation,
            weighted_eval_losses=weighted_eval_losses,
        )

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregate the model deltas.

        Args:
            results (List[Tuple[ClientProxy, FitRes]]): The client identifiers and the results of their local training
                that need to be aggregated on the server-side.
            failures (List[Union[Tuple[ClientProxy, FitRes], BaseException]]): These are the results and exceptions
                from clients that experienced an issue during training, such as timeouts or exceptions.

        Returns:
            Tuple[Optional[Parameters], Dict[str, Scalar]]: The aggregated model weights and the metrics dictionary.
        """
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}
        
        client_model_delta_vectors = [parameters_to_ndarrays(client_response.parameters)[0] for _, client_response in results]
        trainset_size = sum([client_response.num_examples for _, client_response in results])
        client_count = len(results)

        global_model_delta_vector = client_model_delta_vectors.pop(0)

        # sum
        for client_model_delta in client_model_delta_vectors:
            global_model_delta_vector += client_model_delta
        
        # average
        global_model_delta_vector /= client_count
        log(DEBUG, global_model_delta_vector[:10])

        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")


        return global_model_delta_vector, metrics_aggregated, (trainset_size, client_count)