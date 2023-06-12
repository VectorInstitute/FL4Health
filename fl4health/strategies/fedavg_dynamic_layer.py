from logging import WARNING
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from flwr.common import NDArray, NDArrays, Parameters, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.common.logger import log
from flwr.common.typing import FitRes, Scalar
from flwr.server.client_proxy import ClientProxy

from fl4health.strategies.fedavg_sampling import FedAvgSampling


class FedAvgDynamicLayer(FedAvgSampling):
    """
    A generalization of the fedavg strategy where the server can receive any arbitrary subset of the layers from
    any arbitrary subset of the clients, and weighted average for each received layer is performed independently.
    """

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
        names_to_layers = {}
        total_num_examples = {}

        for packed_layers, num_examples in results:
            layers, names = self.unpack_parameters(packed_layers)
            for layer, name in zip(layers, names):
                if name not in names_to_layers:
                    names_to_layers[name] = layer * num_examples
                    total_num_examples[name] = num_examples
                names_to_layers[name] += layer * num_examples
                total_num_examples[name] += num_examples

        name_to_layers_aggregated = {
            name_key: names_to_layers[name_key] / total_num_examples[name_key] for name_key in names_to_layers
        }

        return name_to_layers_aggregated
