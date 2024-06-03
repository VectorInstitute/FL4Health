from functools import reduce
from logging import WARNING
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch.nn as nn
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
from opacus import GradSampleModule

from fl4health.client_managers.base_sampling_manager import BaseFractionSamplingManager
from fl4health.parameter_exchange.parameter_packer import ParameterPackerWithControlVariates
from fl4health.strategies.basic_fedavg import BasicFedAvg
from fl4health.utils.parameter_extraction import get_all_model_parameters

class FLASH(BasicFedAvg):
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
        weighted_eval_losses: bool = True,
        learning_rate: float = 1.0,
        initial_control_variates: Optional[Parameters] = None,
        model: Optional[nn.Module] = None,
        gamma: float = 0.01,
        tau: float = 1.0,
    ) -> None:
        """
        FLASH Federated Learning strategy. Implementation based on https://proceedings.mlr.press/v202/panchal23a/panchal23a.pdf
        Args:
            initial_parameters (Parameters): Initial model parameters to which all client models are set.
            fraction_fit (float, optional): Fraction of clients used during training. Defaults to 1.0.
            fraction_evaluate (float, optional): Fraction of clients used during validation. Defaults to 1.0.
            min_available_clients (int, optional): Minimum number of total clients in the system.
                Defaults to 2.
            evaluate_fn (Optional[
                Callable[[int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]]]
            ]):
                Optional function used for central server-side evaluation. Defaults to None.
            on_fit_config_fn (Optional[Callable[[int], Dict[str, Scalar]]], optional):
                Function used to configure training by providing a configuration dictionary. Defaults to None.
            on_evaluate_config_fn (Optional[Callable[[int], Dict[str, Scalar]]], optional):
               Function used to configure server-side central validation by providing a Config dictionary.
               Defaults to None.
            accept_failures (bool, optional):Whether or not accept rounds containing failures. Defaults to True.
            fit_metrics_aggregation_fn (Optional[MetricsAggregationFn], optional): Metrics aggregation function.
                Defaults to None.
            evaluate_metrics_aggregation_fn (Optional[MetricsAggregationFn], optional): Metrics aggregation function.
                Defaults to None.
            weighted_eval_losses (bool, optional): Determines whether losses during evaluation are linearly weighted
                averages or a uniform average. FedAvg default is weighted average of the losses by client dataset
                counts. Defaults to True.
            learning_rate (float, optional): Learning rate for server side optimization. Defaults to 1.0.
            initial_control_variates (Optional[Parameters], optional): These are the initial set of control variates
                to use for the scaffold strategy both on the server and client sides. It is optional, but if it is not
                provided, the strategy must receive a model that reflects the architecture to be used on the clients.
                Defaults to None.
            model (Optional[nn.Module], optional): If provided and initial_control_variates is not, this is used to
                set the server control variates and the initial control variates on the client side to all zeros.
                If initial_control_variates are provided, they take precedence. Defaults to None.
        """
        self.server_model_weights = parameters_to_ndarrays(initial_parameters)
        initial_control_variates = self.initialize_control_variates(initial_control_variates, model)
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
            weighted_aggregation=False,
            weighted_eval_losses=weighted_eval_losses,
        )
        self.learning_rate = learning_rate
        self.parameter_packer = ParameterPackerWithControlVariates(len(self.server_model_weights))
        self.gamma = gamma
        self.tau = tau
        self.d = np.zeros_like(self.server_model_weights)
        
    def initialize_control_variates(
        self, initial_control_variates: Optional[Parameters], model: Optional[nn.Module]
    ) -> Parameters:
        """
        This is a helper function for the SCAFFOLD strategy init function to initialize the server_control_variates.
        It either initializes the control variates with custom provided variates or using the provided model
        architecture.

        Args:
            initial_control_variates (Optional[Parameters]): These are the initial set of control variates
                to use for the scaffold strategy both on the server and client sides. It is optional, but if it is not
                provided, the strategy must receive a model that reflects the architecture to be used on the clients.
                Defaults to None.
            model (Optional[nn.Module]): If provided and initial_control_variates is not, this is used to
                set the server control variates and the initial control variates on the client side to all zeros.
                If initial_control_variates are provided, they take precedence. Defaults to None.

        Returns:
            Parameters: This quantity represents the initial values for the control variates for the server and on the
            client-side.
        Raises:
            ValueError: This error will be raised if neither a model nor initial control variates are provided.
        """
        if initial_control_variates is not None:
            self.server_control_variates = parameters_to_ndarrays(initial_control_variates)
            return initial_control_variates
        elif model is not None:
            zero_control_variates = [np.zeros_like(val.data) for val in model.parameters() if val.requires_grad]
            self.server_control_variates = zero_control_variates
            return ndarrays_to_parameters(zero_control_variates)
        else:
            raise ValueError("Both initial_control_variates and model are None. One must be defined.")

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}

        updated_params = [parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]
        aggregated_params = self.aggregate(updated_params)

        weights, control_variates_update = self.parameter_packer.unpack_parameters(aggregated_params)

        self.server_model_weights = self.compute_updated_weights(weights)
        self.server_control_variates = self.compute_updated_control_variates(control_variates_update)

        parameters = self.parameter_packer.pack_parameters(self.server_model_weights, self.server_control_variates)

        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return ndarrays_to_parameters(parameters), metrics_aggregated

    def compute_updated_weights(self, weights: NDArrays) -> NDArrays:
        delta_weights = self.compute_parameter_delta(weights, self.server_model_weights)
        server_model_weights = self.compute_updated_parameters(
            self.learning_rate, self.server_model_weights, delta_weights
        )
        return server_model_weights

    def compute_updated_control_variates(self, control_variates_update: NDArrays) -> NDArrays:
        server_control_variates = self.compute_updated_parameters(
            self.fraction_fit, self.server_control_variates, control_variates_update
        )
        return server_control_variates

    def compute_parameter_delta(self, params_1: NDArrays, params_2: NDArrays) -> NDArrays:
        parameter_delta: NDArrays = [param_1 - param_2 for param_1, param_2 in zip(params_1, params_2)]
        return parameter_delta

    def compute_updated_parameters(
        self, scaling_coefficient: float, original_params: NDArrays, parameter_updates: NDArrays
    ) -> NDArrays:
        updated_parameters = [
            original_param + scaling_coefficient * update
            for original_param, update in zip(original_params, parameter_updates)
        ]
        return updated_parameters

    def aggregate(self, params: List[NDArrays]) -> NDArrays:
        num_clients = len(params)
        params_prime: NDArrays = [reduce(np.add, layer_updates) / num_clients for layer_updates in zip(*params)]
        return params_prime

    def configure_fit_all(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        assert isinstance(client_manager, BaseFractionSamplingManager)

        config = {}
        if self.on_fit_config_fn is not None:
            config = self.on_fit_config_fn(server_round)

        fit_ins = FitIns(parameters, config)
        clients = client_manager.sample_all(self.min_available_clients)

        return [(client, fit_ins) for client in clients]