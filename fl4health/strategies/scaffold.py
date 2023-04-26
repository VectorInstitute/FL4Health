from functools import reduce
from logging import WARNING
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from flwr.common import MetricsAggregationFn, NDArrays, Parameters, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.common.logger import log
from flwr.common.typing import FitRes, Scalar
from flwr.server.client_proxy import ClientProxy

from fl4health.strategies.fedavg_sampling import FedAvgSampling


class Scaffold(FedAvgSampling):

    # pylint: disable=too-many-arguments,too-many-instance-attributes
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
        learning_rate: Optional[float] = 1.0
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
        initial_parameters : Parameters, optional
            Initial global model parameters.
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn]
            Metrics aggregation function, optional.
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn]
            Metrics aggregation function, optional.
        learning_rate: Optional[float]
            Learning rate for server side optimization.
        """
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
        self.server_model_weights = parameters_to_ndarrays(initial_parameters)
        self.server_control_variates = [np.zeros_like(arr) for arr in self.server_model_weights]

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

        # Convert results
        updated_params = [(parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples) for _, fit_res in results]

        # x = 1 / |S| * sum(x_i) and c = 1 / |S| * sum(c_i)
        aggregated_params = self.aggregate(updated_params)

        weights, control_variates_update = self.unpack_parameters(aggregated_params)

        # x_update = y_i - x
        delta_weights: NDArrays = [
            updated_weight - current_weight
            for updated_weight, current_weight in zip(weights, self.server_model_weights)
        ]

        # x = x + lr * x_update
        self.server_model_weights = [
            current_weight + self.learning_rate * delta_weight
            for current_weight, delta_weight in zip(self.server_model_weights, delta_weights)
        ]

        # c = c + |S| / N * c_update
        self.server_control_variates = [
            current_control_variate + self.fraction_fit * control_variate_update
            for current_control_variate, control_variate_update in zip(
                self.server_control_variates, control_variates_update
            )
        ]

        parameters = self.pack_parameters(self.server_model_weights, self.server_control_variates)

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return ndarrays_to_parameters(parameters), metrics_aggregated

    def unpack_parameters(self, parameters: NDArrays) -> Tuple[NDArrays, NDArrays]:
        """
        Split params into model_weights and control_variates.
        """
        split_size = len(parameters) // 2
        model_weights, control_variates = parameters[:split_size], parameters[split_size:]
        return model_weights, control_variates

    def pack_parameters(self, model_weights: NDArrays, control_variates: NDArrays) -> NDArrays:
        """
        Extends parameter list to include model weights and server control variates.
        """
        return model_weights + control_variates

    def aggregate(self, results: List[Tuple[NDArrays, int]]) -> NDArrays:
        """
        Simple unweighted average to aggregate params. Consistent with paper.
        """
        num_clients = len(results)

        weights = [weights for weights, _ in results]

        # Compute average weights of each layer
        weights_prime: NDArrays = [reduce(np.add, layer_updates) / num_clients for layer_updates in zip(*weights)]

        return weights_prime
