from functools import reduce
from logging import WARNING
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from flwr.common import (
    FitIns,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.common.typing import FitRes, Scalar
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from fl4health.client_managers.base_sampling_manager import BaseSamplingManager
from fl4health.parameter_exchange.parameter_packer import ParameterPackerWithControlVariates
from fl4health.strategies.fedavg_sampling import FedAvgSampling


class Scaffold(FedAvgSampling):
    """
    Strategy for Scaffold algorithm as specified in https://arxiv.org/abs/1910.06378
    """

    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
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
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        learning_rate: float = 1.0,
        initial_control_variates: Parameters,
    ) -> None:
        """Scaffold Federated Learning strategy.

        Implementation based on https://arxiv.org/pdf/1910.06378.pdf

        Parameters
        ----------
        fraction_fit : float, optional
            Fraction of clients used during training. Defaults to 1.0.
        fraction_evaluate : float, optional
            Fraction of clients used during validation. Defaults to 1.0.
        min_available_clients : int, optional
            Minimum number of total clients in the system. Defaults to 2.
        evaluate_fn : Optional[
            Callable[
                [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]]
            ]
        ]
            Optional function used for validation. Defaults to None.
        on_fit_config_fn : Callable[[int], Dict[str, Scalar]], optional
            Function used to configure training. Defaults to None.
        on_evaluate_config_fn : Callable[[int], Dict[str, Scalar]], optional
            Function used to configure validation. Defaults to None.
        accept_failures : bool, optional
            Whether or not accept rounds containing failures. Defaults to True.
        initial_parameters : Parameters
            Initial global model parameters.
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn]
            Metrics aggregation function, optional.
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn]
            Metrics aggregation function, optional.
        learning_rate: Optional[float]
            Learning rate for server side optimization.
        """
        self.server_model_weights = parameters_to_ndarrays(initial_parameters)
        self.server_control_variates = parameters_to_ndarrays(initial_control_variates)
        initial_parameters.tensors.extend(initial_control_variates.tensors)
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        )
        self.learning_rate = learning_rate
        self.parameter_packer = ParameterPackerWithControlVariates(len(self.server_model_weights))

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Convert results with packed params of model weights and client control variate updates
        updated_params = [parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]

        # x = 1 / |S| * sum(x_i) and c = 1 / |S| * sum(delta_c_i)
        # Aggregation operation over packed params (includes both weights and control variate updates)
        aggregated_params = self.aggregate(updated_params)

        weights, control_variates_update = self.parameter_packer.unpack_parameters(aggregated_params)

        self.server_model_weights = self.compute_updated_weights(weights)
        self.server_control_variates = self.compute_updated_control_variates(control_variates_update)

        parameters = self.parameter_packer.pack_parameters(self.server_model_weights, self.server_control_variates)

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return ndarrays_to_parameters(parameters), metrics_aggregated

    def compute_parameter_delta(self, params_1: NDArrays, params_2: NDArrays) -> NDArrays:
        """
        Computes elementwise difference of two lists of NDarray
        where elements in params_2 are subtracted from elements in params_1
        """
        parameter_delta: NDArrays = [param_1 - param_2 for param_1, param_2 in zip(params_1, params_2)]

        return parameter_delta

    def compute_updated_parameters(
        self, scaling_coefficient: float, original_params: NDArrays, parameter_updates: NDArrays
    ) -> NDArrays:
        """
        Computes updated_params by moving in the direction of parameter_updates
        with a step proportional the scaling coefficient.
        """

        updated_parameters = [
            original_param + scaling_coefficient * update
            for original_param, update in zip(original_params, parameter_updates)
        ]

        return updated_parameters

    def aggregate(self, params: List[NDArrays]) -> NDArrays:
        """
        Simple unweighted average to aggregate params. Consistent with paper.
        """
        num_clients = len(params)

        # Compute average weights of each layer
        params_prime: NDArrays = [reduce(np.add, layer_updates) / num_clients for layer_updates in zip(*params)]

        return params_prime

    def configure_fit_all(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        # This strategy requires the client manager to be of type at least BaseSamplingManager
        assert isinstance(client_manager, BaseSamplingManager)
        """Configure the next round of training."""
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)

        fit_ins = FitIns(parameters, config)

        clients = client_manager.sample_all(self.min_available_clients)

        # Return client/config pairs
        return [(client, fit_ins) for client in clients]

    def compute_updated_weights(self, weights: NDArrays) -> NDArrays:
        # x_update = y_i - x
        delta_weights = self.compute_parameter_delta(weights, self.server_model_weights)

        # x = x + lr * x_update
        server_model_weights = self.compute_updated_parameters(
            self.learning_rate, self.server_model_weights, delta_weights
        )

        return server_model_weights

    def compute_updated_control_variates(self, control_variates_update: NDArrays) -> NDArrays:
        # c = c + |S| / N * c_update
        server_control_variates = self.compute_updated_parameters(
            self.fraction_fit, self.server_control_variates, control_variates_update
        )

        return server_control_variates
