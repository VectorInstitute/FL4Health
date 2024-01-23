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


class SecureAggregationStrategy(BasicFedAvg):
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

    def package_request(self, request: Dict[str, Scalar], event_name: str, client_manager: ClientManager) -> Requests:
        # get all online clients for SecAgg (substitute for different client sampler for SecAgg+)
        if isinstance(client_manager, BaseFractionSamplingManager):
            clients_list = client_manager.sample_all(min_num_clients=self.min_available_clients)
        else:
            # Grab all available clients using the basic Flower client manager
            num_available_clients = client_manager.num_available()
            clients_list = client_manager.sample(num_available_clients, min_num_clients=self.min_available_clients)

        # adjoin event_name
        req = {"event_name": event_name, **request}

        # Package dictionary to Flower request format
        wrapper = GetPropertiesIns(req)

        packaged = map(lambda client: (client, wrapper), clients_list)
        return list(packaged)

    def package_single_client_request(
        self, client: ClientProxy, request: Dict[str, Scalar], event_name: str
    ) -> Request:
        # adjoin event_name
        req = {"event_name": event_name, **request}

        # Package dictionary to Flower request format
        wrapper = GetPropertiesIns(req)

        return client, wrapper

    """
    Customize: first compute sum, then communicate w/clients to remove masks, then average
    """

    def sum_results(results: List[Tuple[NDArrays, int]]) -> NDArrays:
        """
        Sum the client parameter vectors.
        """

        weighted_weights = [[layer for layer in weights] for weights, _ in results]

        # Compute unweighted average by summing up across clients for each layer.
        return [reduce(np.add, layer_updates) for layer_updates in zip(*weighted_weights)]

    def get_client_models(self, results: List[Tuple[ClientProxy, FitRes]]) -> List[NDArrays]:

        # unpack parameters
        serialized_models = [client.parameters for _, client in results]

        # deserialize
        client_models = map(lambda serialized_model: parameters_to_ndarrays(serialized_model), serialized_models)

        # This is an array. Each index corresponds a client model.
        # The type at each index is an array of numpy arrays.
        models: List[NDArrays] = list(client_models)
        # self.debugger('model', models)
        return models

    def aggregate_sum(self, client_models: List[NDArrays]):

        # # hard coded modulus
        # MODULUS = 1 << 30

        # number of layers per model
        N = len(client_models[0])

        model_sum = []
        for layer_i in range(N):
            layer_sum = reduce(np.add, [client_model[layer_i] for client_model in client_models])

            # this can be cutomized depending on post processing procedure
            layer_sum = layer_sum  #% MODULUS

            model_sum.append(layer_sum)

        # do post processing (such as taking peudo-Kashin inverse, or averaging)
        # layer_sum /= 3

        return model_sum

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
        arithmetic_modulus: int = 1,
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        """
        Aggregate the results (with modular arithmetic) from the federated fit round. This is done with either weighted
        or unweighted FedAvg, depending on the settings used for the strategy.

        Args:
            server_round (int): Indicates the server round we're currently on.
            arithmetic_modulus (int): modulus for secure aggregation server procedure
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
        
        local_models = [parameters_to_ndarrays(client_response.parameters) for _, client_response in results]
        aggregate_data_size = sum([client_response.num_examples for _, client_response in results])
        log(INFO, f'Training data size: {aggregate_data_size}')

        # initalized to model layers of the first client
        global_model_layers = local_models.pop(0)

        # if server_round == 0:
        #     return ndarrays_to_parameters(global_model_layers), {}

        num_layers = len(global_model_layers)

        # NOTE secure aggregation (mask is removed in summation)
        for local_model_layers in local_models:
            for k in range(num_layers):
                global_model_layers[k] += local_model_layers[k]
                global_model_layers[k] %= arithmetic_modulus
        
        log(DEBUG, f'round {server_round}')
        log(DEBUG, global_model_layers)

        parameters_aggregated = ndarrays_to_parameters(global_model_layers)

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        # TODO this is a work around for sampling clients on round 0
        if server_round == 0:
            return parameters_aggregated, metrics_aggregated

        return parameters_aggregated, metrics_aggregated, aggregate_data_size

    def debugger(self, *info):
        log(DEBUG, 6 * "\n")
        for item in info:
            log(DEBUG, item)


class ServerProprocessor:
    def __init__(self):
        pass


class DiscreteGaussianProcessing:
    def __init__(self):
        pass


class ServerDP:
    def __init__(self):
        pass
