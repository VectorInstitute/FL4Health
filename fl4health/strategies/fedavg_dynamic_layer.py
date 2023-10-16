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

from fl4health.parameter_exchange.parameter_packer import ParameterPackerWithLayerNames
from fl4health.strategies.basic_fedavg import BasicFedAvg


class FedAvgDynamicLayer(BasicFedAvg):
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
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        weighted_aggregation: bool = True,
        weighted_eval_losses: bool = True,
    ) -> None:
        """
        A generalization of the FedAvg strategy where the server can receive any arbitrary subset of the layers from
        any arbitrary subset of the clients, and weighted average for each received layer is performed independently.

        Args:
            fraction_fit (float, optional): Fraction of clients used during training. Defaults to 1.0. Defaults to 1.0.
            fraction_evaluate (float, optional): Fraction of clients used during validation. Defaults to 1.0.
            min_available_clients (int, optional): Minimum number of clients used during validation. Defaults to 2.
            evaluate_fn (Optional[
                Callable[[int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]]]
            ]):
                Optional function used for central server-side evaluation. Defaults to None.
            on_fit_config_fn (Optional[Callable[[int], Dict[str, Scalar]]], optional):
                Function used to configure training by providing a configuration dictionary. Defaults to None.
            on_evaluate_config_fn (Optional[Callable[[int], Dict[str, Scalar]]], optional):
                Function used to configure server-side central validation by providing a Config dictionary.
                Defaults to None.
            accept_failures (bool, optional): Whether or not accept rounds containing failures. Defaults to True.
            initial_parameters (Optional[Parameters], optional): Initial global model parameters. Defaults to None.
            fit_metrics_aggregation_fn (Optional[MetricsAggregationFn], optional): Metrics aggregation function.
                Defaults to None.
            evaluate_metrics_aggregation_fn (Optional[MetricsAggregationFn], optional): Metrics aggregation function.
                Defaults to None.
            weighted_aggregation (bool, optional): Determines whether parameter aggregation is a linearly weighted
                average or a uniform average. FedAvg default is weighted average by client dataset counts.
                Defaults to True.
            weighted_eval_losses (bool, optional): Determines whether losses during evaluation are linearly weighted
                averages or a uniform average. FedAvg default is weighted average of the losses by client dataset
                counts. Defaults to True.
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
            weighted_aggregation=weighted_aggregation,
            weighted_eval_losses=weighted_eval_losses,
        )
        self.parameter_packer = ParameterPackerWithLayerNames()

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregate the results from the federated fit round. The aggregation requires some special treatment, as the
        participating clients are allowed to exchange an arbitrary set of weights. So before aggregation takes place
        alignment must be done using the layer names packed in along with the weights in the client results.

        Args:
            server_round (int): Indicates the server round we're currently on.
            results (List[Tuple[ClientProxy, FitRes]]): The client identifiers and the results of their local training
                that need to be aggregated on the server-side. In this scheme, the clients pack the layer weights into
                the results object along with the weight values to allow for alignment during aggregation.
            failures (List[Union[Tuple[ClientProxy, FitRes], BaseException]]): These are the results and exceptions
                from clients that experienced an issue during training, such as timeouts or exceptions.

        Returns:
            Tuple[Optional[Parameters], Dict[str, Scalar]]: The aggregated model weights and the metrics dictionary.
                For dynamic layer exchange we also pack in the names of all of the layers that were aggregated in this
                phase to allow client's to insert the values into the proper areas of their models.
        """
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

        parameters = self.parameter_packer.pack_parameters(weights, weights_names)

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return ndarrays_to_parameters(parameters), metrics_aggregated

    def aggregate(self, results: List[Tuple[NDArrays, int]]) -> Dict[str, NDArray]:
        """
        Aggregate the different layers across clients that have contributed to a layer. This aggregation may be
        weighted or unweighted. The called functions handle layer alignment.

        Args:
            results (List[Tuple[NDArrays, int]]): The weight results from each client's local training that need to be
                aggregated on the server-side and the number of training samples held on each clien. In this scheme,
                the clients pack the layer weights into the results object along with the weight values to allow for
                alignment during aggregation.

        Returns:
            Dict[str, NDArray]: A dictionary mapping the name of the layer that was aggregated to the aggregated
                weights.
        """
        if self.weighted_aggregation:
            return self.weighted_aggregate(results)
        else:
            return self.unweighted_aggregate(results)

    def weighted_aggregate(self, results: List[Tuple[NDArrays, int]]) -> Dict[str, NDArray]:
        """
        Results consists of the layer weights (and their names) sent by clients who participated in this round of
        training. Since each client can send an arbitrary subset of layers, the aggregate performs weighted averaging
        for each layer separately.

        Args:
            results (List[Tuple[NDArrays, int]]): The weight results from each client's local training that need to be
                aggregated on the server-side and the number of training samples held on each clien. In this scheme,
                the clients pack the layer weights into the results object along with the weight values to allow for
                alignment during aggregation.

        Returns:
            Dict[str, NDArray]: A dictionary mapping the name of the layer that was aggregated to the aggregated
                weights.
        """
        names_to_layers: DefaultDict[str, List[NDArray]] = defaultdict(list)
        total_num_examples: DefaultDict[str, int] = defaultdict(int)

        for packed_layers, num_examples in results:
            layers, names = self.parameter_packer.unpack_parameters(packed_layers)
            for layer, name in zip(layers, names):
                names_to_layers[name].append(layer * num_examples)
                total_num_examples[name] += num_examples

        name_to_layers_aggregated = {
            name_key: reduce(np.add, names_to_layers[name_key]) / total_num_examples[name_key]
            for name_key in names_to_layers
        }

        return name_to_layers_aggregated

    def unweighted_aggregate(self, results: List[Tuple[NDArrays, int]]) -> Dict[str, NDArray]:
        """
        Results consists of the layer weights (and their names) sent by clients who participated in this round of
        training. Since each client can send an arbitrary subset of layers, the aggregate performs uniform averaging
        for each layer separately.

        Args:
            results (List[Tuple[NDArrays, int]]): The weight results from each client's local training that need to be
                aggregated on the server-side and the number of training samples held on each clien. In this scheme,
                the clients pack the layer weights into the results object along with the weight values to allow for
                alignment during aggregation.

        Returns:
            Dict[str, NDArray]: A dictionary mapping the name of the layer that was aggregated to the aggregated
                weights.
        """
        names_to_layers: DefaultDict[str, List[NDArray]] = defaultdict(list)
        total_num_clients: DefaultDict[str, int] = defaultdict(int)

        for packed_layers, _ in results:
            layers, names = self.parameter_packer.unpack_parameters(packed_layers)
            for layer, name in zip(layers, names):
                names_to_layers[name].append(layer)
                total_num_clients[name] += 1

        name_to_layers_aggregated = {
            name_key: reduce(np.add, names_to_layers[name_key]) / total_num_clients[name_key]
            for name_key in names_to_layers
        }

        return name_to_layers_aggregated
