from logging import WARNING
from typing import Callable, Dict, List, Optional, Tuple, Union

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

from fl4health.strategies.fedavg_sampling import FedAvgSampling


class FedAvgDynamicLayer(FedAvgSampling):
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
    ) -> None:
        """
        A generalization of the fedavg strategy where the server can receive any arbitrary subset of the layers from
        any arbitrary subset of the clients, and weighted average for each received layer is performed independently.

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
        # self.learning_rate = learning_rate
        # self.server_model_weights = parameters_to_ndarrays(initial_parameters)

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

        # Convert client layer weights and names into ndarrays
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples) for _, fit_res in results
        ]

        # For each layer of the model, perform weighted average of all received weights from clients
        aggregated_params = self.aggregate(weights_results)

        weights_names = []
        weights = []
        for name in aggregated_params:
            weights_names.append(name)
            weights.append(aggregated_params[name])

        parameters = self.pack_parameters(weights, weights_names)

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return ndarrays_to_parameters(parameters), metrics_aggregated

    def unpack_parameters(self, parameters: NDArrays) -> Tuple[NDArrays, List[str]]:
        """
        Split params into model_weights and weights_names.
        """
        split_size = len(parameters) - 1
        model_parameters = parameters[:split_size]
        param_names = parameters[split_size:][0].tolist()
        return model_parameters, param_names

    def pack_parameters(self, model_weights: NDArrays, weights_names: List[str]) -> NDArrays:
        """
        Extends parameter list to include model weights and their names.
        """
        return model_weights + [np.array(weights_names)]

    def aggregate(self, results: List[Tuple[NDArrays, int]]) -> Dict[str, NDArray]:
        """
        Results consists of the layer weights (and their names) sent by clients
        who participated in this round of training.
        Since each client can send an arbitrary subset of layers,
        the aggregate performs weighted averaging for each layer separately.
        """
        # Calculate the total number of examples used during training
        # and unpack layer weights with their corresponding names
        num_examples_total = sum([num_examples for _, num_examples in results])

        name_to_weights = {}

        for packed_weights, num_examples in results:
            weights, names = self.unpack_parameters(packed_weights)
            for weight, name in zip(weights, names):
                if name not in name_to_weights:
                    name_to_weights[name] = weight * num_examples / num_examples_total
                else:
                    name_to_weights[name] += weight * num_examples / num_examples_total

        return name_to_weights
