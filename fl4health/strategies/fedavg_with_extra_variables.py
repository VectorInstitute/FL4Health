from collections import defaultdict
from functools import reduce
from logging import WARNING
from typing import Callable, DefaultDict, Dict, List, Optional, Tuple, Union

import numpy as np
from flwr.common import (
    MetricsAggregationFn,
    NDArray,
    NDArrays,
    Parameters,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.common.typing import FitRes, Scalar
from flwr.server.client_proxy import ClientProxy

from fl4health.parameter_exchange.parameter_packer import ParameterPackerWithExtraVariables
from flwr.server.strategy import FedAvg


class FedAvgWithExtraVariables(FedAvg):
    """
    A generalization of the fedavg strategy where the server can receive any extra vriables and their name from
    the clients, and weighted average for each received layer is performed independently.
    """

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
        initial_extra_variables: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
    ) -> None:
        
        self.server_model_weights = parameters_to_ndarrays(initial_parameters)
        self.server_extra_variables = parameters_to_ndarrays(initial_extra_variables)
        initial_parameters.tensors.extend(initial_extra_variables.tensors)
        self.server_model_weights = parameters_to_ndarrays(initial_parameters)
        super().__init__(
            fraction_fit = fraction_fit,
            fraction_evaluate = fraction_evaluate,
            min_fit_clients = min_fit_clients,
            min_evaluate_clients = min_evaluate_clients,
            min_available_clients = min_available_clients,
            evaluate_fn = evaluate_fn,
            on_fit_config_fn = on_fit_config_fn,
            on_evaluate_config_fn = on_evaluate_config_fn,
            accept_failures = accept_failures,
            initial_parameters = initial_parameters,
            fit_metrics_aggregation_fn = fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn,
        )
        self.parameter_packer = ParameterPackerWithExtraVariables(len(self.server_model_weights))
        self.variables = [0.0]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        
        parameters_aggregated, metrics_aggregated = super().aggregate_fit(server_round, results, failures)
        parameters = self.parameter_packer.pack_parameters(parameters_to_ndarrays(parameters_aggregated), self.server_extra_variables)
        return ndarrays_to_parameters(parameters), metrics_aggregated
    
    def set_variable(
        self,
        extra_variables: List[float],
    ) -> None:
        self.server_extra_variables = extra_variables
        return 

