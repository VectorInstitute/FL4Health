from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from flwr.common import MetricsAggregationFn, NDArrays, Parameters, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.common.typing import FitRes, Scalar
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from fl4health.parameter_exchange.parameter_packer import ParameterPackerWithExtraVariables


class FedAvgWithExtraVariables(FedAvg):
    """
    A generalization of the fedavg strategy where the server can receive any extra vriables as
    in a dictionary formt where key is the name of the variable and value is the parameter value of the variable.
    Server can update the extra variables and send it back to the clients.
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
        initial_parameters: Parameters,
        initial_extra_variables: Dict[str, Parameters],
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
    ) -> None:

        self.server_model_weights = parameters_to_ndarrays(initial_parameters)
        self.server_extra_variables = {}
        initial_parameters.tensors.extend(
            ndarrays_to_parameters([np.array(list(initial_extra_variables.keys()))]).tensors
        )

        for key, value in initial_extra_variables.items():
            self.server_extra_variables[key] = parameters_to_ndarrays(value)
            initial_parameters.tensors.extend(value.tensors)

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
        )
        self.parameter_packer = ParameterPackerWithExtraVariables(len(self.server_model_weights))

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        # get the aggregated parameters and metrics
        parameters_aggregated, metrics_aggregated = super().aggregate_fit(server_round, results, failures)

        # if aggregation is successful, pack updated extra vriables in the server to the parameters with model wieght
        if parameters_aggregated is not None:
            parameters = self.parameter_packer.pack_parameters(
                parameters_to_ndarrays(parameters_aggregated), self.server_extra_variables
            )
            return ndarrays_to_parameters(parameters), metrics_aggregated
        else:
            return parameters_aggregated, metrics_aggregated

    def set_extra_variables(
        self,
        keys: List[str],
        values: List[NDArrays],
    ) -> None:
        for key, value in zip(keys, values):
            self.server_extra_variables[key] = value
        return
