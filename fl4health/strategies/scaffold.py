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

from fl4health.client_managers.base_sampling_manager import BaseFractionSamplingManager
from fl4health.parameter_exchange.parameter_packer import ParameterPackerWithControlVariates
from fl4health.strategies.basic_fedavg import BasicFedAvg


class Scaffold(BasicFedAvg):
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
    ) -> None:
        """
        Scaffold Federated Learning strategy. Implementation based on https://arxiv.org/pdf/1910.06378.pdf
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
                If initial_control_variates are provided, they take precendence. Defaults to None.
        """

        self.server_model_weights = parameters_to_ndarrays(initial_parameters)
        # Setup the initial control variates on the server-side and store them to be transmitted to the clients
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
                If initial_control_variates are provided, they take precendence. Defaults to None.

        Returns:
            Parameters: This quantity represents the initial values for the control variates for the server and on the
            client-side.
        Raises:
            ValueError: This error will be raised if neither a model nor initial control variates are provided.
        """
        if initial_control_variates is not None:
            # If we've been provided with a set of initial control variates, we use those values
            self.server_control_variates = parameters_to_ndarrays(initial_control_variates)
            return initial_control_variates
        elif model is not None:
            # If no initial values are provided but a model structure has been given, we initialize the control
            # variates to zeros as recommended in the SCAFFOLD paper.
            zero_control_variates = [np.zeros_like(val.data) for val in model.parameters() if val.requires_grad]
            self.server_control_variates = zero_control_variates
            return ndarrays_to_parameters(zero_control_variates)
        else:
            # Either a model structure or custom initial values for the control variates must be provided to run
            # SCAFFOLD
            raise ValueError(
                "Both initial_control_variates and model are None. One must be defined in order to establish "
                "initial values for the control variates."
            )

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Performs server-side aggregation of model weights and control variates associated with the SCAFFOLD method
        Both model weights and control variates are aggregated through UNWEIGHTED averaging consistent with the paper.
        The newly aggregated weights and control variates are then repacked and sent back to the clients.

        This function also handles aggregation of training run metrics (i.e. accuracy over the local training etc.)
        through the fit_metrics_aggregation_fn provided in constructing the strategy.

        Args:
            server_round (int): What round of FL we're on (from servers perspective).
            results (List[Tuple[ClientProxy, FitRes]]): These are the "successful" training run results. By default
                these results are the only ones used in aggregation, even if some of the failed clients have partial
                results (in the failures list).
            failures (List[Union[Tuple[ClientProxy, FitRes], BaseException]]): This is the list of clients that
                "failed" during the training phase for one reason or another, including timeouts and exceptions.

        Returns:
            Tuple[Optional[Parameters], Dict[str, Scalar]]: The aggregated weighted and metrics dictionary. The
                parameters are optional and will be none in the even that there are no successful clients or there
                were failures and they are not accepted.
        """
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
        Computes elementwise difference of two lists of NDarray where elements in params_2 are subtracted from
        elements in params_1.

        Args:
            params_1 (NDArrays): Parameters to be subtracted from.
            params_2 (NDArrays): Parameters to subtract from params_1.

        Returns:
            NDArrays: Elementwise subtraction result across all numpy arrays.
        """
        parameter_delta: NDArrays = [param_1 - param_2 for param_1, param_2 in zip(params_1, params_2)]

        return parameter_delta

    def compute_updated_parameters(
        self, scaling_coefficient: float, original_params: NDArrays, parameter_updates: NDArrays
    ) -> NDArrays:
        """
        Computes updated_params by moving in the direction of parameter_updates with a step proportional the scaling
        coefficient. Calculates original_params + scaling_coefficient * parameter_updates.

        Args:
            scaling_coefficient (float): Scaling length for the parameter updates (can be thought of as
                "learning rate").
            original_params (NDArrays): parameters to be updated.
            parameter_updates (NDArrays): update direction to update the original_params.

        Returns:
            NDArrays: Updated numpy arrays according to original_params + scaling_coefficient * parameter_updates.
        """

        updated_parameters = [
            original_param + scaling_coefficient * update
            for original_param, update in zip(original_params, parameter_updates)
        ]

        return updated_parameters

    def aggregate(self, params: List[NDArrays]) -> NDArrays:
        """
        Simple unweighted average to aggregate params, consistent with SCAFFOLD paper. This is "element-wise"
        averaging.

        Args:
            params (List[NDArrays]): numpy arrays whose entries are to be averaged together.

        Returns:
            NDArrays: elementwise average over the list of numpy arrays.
        """
        num_clients = len(params)

        # Compute average weights of each layer
        params_prime: NDArrays = [reduce(np.add, layer_updates) / num_clients for layer_updates in zip(*params)]

        return params_prime

    def configure_fit_all(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """
        This function configures ALL clients for a training round. That is, it forces the client manager to grab all
        of the available clients to participate in the training round. By default, the manager will at least wait for
        the min_available_clients threshold to be met. Thereafter it will simply grab all available clients for
        participation.

        The function follows the standard configuration flow where the on_fit_config_fn function is used to produce
        configurations to be sent to all clients. These are packaged with the provided parameters and set over to the
        clients.

        Args:
            server_round (int): Indicates the server round we're currently on.
            parameters (Parameters): The parameters to be used to initialize the clients for the fit round.
            client_manager (ClientManager): The manager used to grab all of the clients. Currently we restrict this to
                be BaseFractionSamplingManager, which has a "sample all" function built in.

        Returns:
            List[Tuple[ClientProxy, FitIns]]: List of sampled client identifiers and the configuration/parameters to
                be sent to each client (packaged as FitIns).
        """

        # This strategy requires the client manager to be of type at least BaseFractionSamplingManager
        assert isinstance(client_manager, BaseFractionSamplingManager)

        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)

        fit_ins = FitIns(parameters, config)

        clients = client_manager.sample_all(self.min_available_clients)

        # Return client/config pairs
        return [(client, fit_ins) for client in clients]

    def compute_updated_weights(self, weights: NDArrays) -> NDArrays:
        """
        Computes and update to the current self.server_model_weights. This assumes that the weights represents the
        raw weights aggregated from the client. Therefore it first needs to be turned into a "delta" with
                weights - self.server_model_weights.
        Then this is used to update with a learning rate scalar (set by self.learning_rate) as
                self.server_model_weights + self.learning_rate * (weights - self.server_model_weights).

        Args:
            weights (NDArrays): The updated weights (aggregated from the clients).

        Returns:
            NDArrays: self.server_model_weights + self.learning_rate * (weights - self.server_model_weights)
                These are the updated server model weights.
        """
        # x_update = y_i - x
        delta_weights = self.compute_parameter_delta(weights, self.server_model_weights)

        # x = x + lr * x_update
        server_model_weights = self.compute_updated_parameters(
            self.learning_rate, self.server_model_weights, delta_weights
        )

        return server_model_weights

    def compute_updated_control_variates(self, control_variates_update: NDArrays) -> NDArrays:
        """
        Given the aggregated control variates from the clients, this updates the server control variates in line with
        the paper. If c is the server control variates and c_update is the client control variates, then this update
        takes the form
                c + |S| / N * c_update,
        where |S| is the number of clients that participated and N is the total number of clients |S|/N is the
        proportion given by fraction fit.

        Args:
            control_variates_update (NDArrays): Aggregated control variates received from the clients
                (uniformly averaged).

        Returns:
            NDArrays: Updated server control variates according to the formula.
        """
        # c = c + |S| / N * c_update
        server_control_variates = self.compute_updated_parameters(
            self.fraction_fit, self.server_control_variates, control_variates_update
        )

        return server_control_variates
