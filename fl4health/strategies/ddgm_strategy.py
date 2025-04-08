from logging import DEBUG, INFO, WARNING
from typing import Callable, Dict, List, Optional, Tuple, Union
from functools import reduce
import numpy as np
import torch

from flwr.common import (
    parameters_to_ndarrays,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetPropertiesIns,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from fl4health.client_managers.base_sampling_manager import BaseFractionSamplingManager
from fl4health.strategies.aggregate_utils import aggregate_results
from fl4health.strategies.basic_fedavg import BasicFedAvg

from fl4health.privacy_mechanisms.discrete_gaussian_mechanism import (
    generate_discrete_gaussian_vector,
    fwht,
    shift_transform
)

Requests = List[Tuple[ClientProxy, GetPropertiesIns]]
Request = Tuple[ClientProxy, GetPropertiesIns]

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class DDGMStrategy(BasicFedAvg):
    """
    DDGM strategy for federated learning.
    """
    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: (
            Callable[[int, NDArrays, dict[str, Scalar]], tuple[float, dict[str, Scalar]] | None] | None
        ) = None,
        on_fit_config_fn: Callable[[int], dict[str, Scalar]] | None = None,
        on_evaluate_config_fn: Callable[[int], dict[str, Scalar]] | None = None,
        accept_failures: bool = True,
        initial_parameters: Parameters | None = None,
        fit_metrics_aggregation_fn: MetricsAggregationFn | None = None,
        evaluate_metrics_aggregation_fn: MetricsAggregationFn | None = None,
        weighted_aggregation: bool = True,
        weighted_eval_losses: bool = True,
        ddgm_config: dict[str, float] = None,
    ) -> None:
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
            weighted_aggregation = weighted_aggregation,
            weighted_eval_losses = weighted_eval_losses
        )
        self.server_model_weights = parameters_to_ndarrays(initial_parameters)
        
        self.granularity = ddgm_config['granularity']
        self.model_dim = ddgm_config['model_dim']
        
    def aggregate_modular_sum(self, params: list[NDArrays], arithmetic_modulus: int) -> NDArrays:
        """
        Aggregate the parameters from the clients.
        """
        num_clients = len(params)

        # Compute sum of weights of each layer
        params_prime: NDArrays = [reduce(np.add, layer_updates) for layer_updates in zip(*params)]
        
        # Apply modular arithmetic
        params_prime = [param % arithmetic_modulus for param in params_prime]
        
        return params_prime
    
    
    def aggregate_sum(self, params: list[NDArrays]) -> NDArrays:
        """
        Aggregate the parameters from the clients.
        """
        num_clients = len(params)

        # Compute sum of weights of each layer
        params_prime: NDArrays = [reduce(np.add, layer_updates) for layer_updates in zip(*params)]
        
        return params_prime
    
    def aggregate_avg(self, params: list[NDArrays]) -> NDArrays:
        """
        Aggregate the parameters from the clients.
        """
        num_clients = len(params)

        # Compute average weights of each layer
        params_prime: NDArrays = [reduce(np.add, layer_updates) / num_clients for layer_updates in zip(*params)]
        
        return params_prime
    
    def ddgm_server_aggregation(deltas: list[NDArrays], arithmetic_modulus: int, sign_vector: torch.Tensor) -> NDArrays:
        """
        Aggregate the parameters from the clients.
        """
        num_clients = len(deltas)
        
        # aggregate (sum) the results
        aggregated_deltas = self.aggregate_modular_sum(deltas, arithmetic_modulus) 
        
        vectorized_aggregated_deltas = vectorize_model(aggregated_deltas)
        
        # server procedure
        vector = shift_transform(vectorized_aggregated_deltas, arithmetic_modulus)
        
        assert arithmetic_modulus % 2 == 0
        bound = arithmetic_modulus // 2
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        vector = torch.from_numpy(vector).to(device=device)
        vector = fwht(vector)
        vector = torch.mul(
            self.granularity * sign_vector,
            vector
        )
        
        vector = vector[:self.model_dim] 
        vector /= num_clients

        return vector.cpu().numpy()
        
    
    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
        arithmetic_modulus: int,
        sign_vector: torch.Tensor,
    ) -> tuple[Parameters | None, dict[str, Scalar]]:
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}
        
        #  process the results
        deltas = [parameters_to_ndarrays(client_response.parameters) for _, client_response in results]
        
        # apply the DDGM server aggregation
        recovered_delta_avged = self.ddgm_server_aggregation(deltas, arithmetic_modulus)
        
        self.server_model = unvectorize_model(self.checkpoint_and_state_module.model, recovered_delta_avged)
        self.revert_layer_dtype()
        self.parameters = ndarrays_to_parameters(
            [layer.cpu().numpy() for layer in self.server_model.state_dict().values()]
        )
        
        updated_model_parameters = [original_param + update for original_param, update in zip(self.server_model_weights, aggregated_deltas)]
        
        
        
        self.server_model_weights = updated_model_parameters
        
        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return ndarrays_to_parameters(updated_model_parameters), metrics_aggregated
    
    def configure_fit(
        self, 
        server_round: int, 
        parameters: Parameters, 
        client_manager: ClientManager,
        arithmetic_modulus: int,
        # sign_vector: torch.Tensor,
    ) -> list[tuple[ClientProxy, FitIns]]:
        """
        Add the sign vector and the arithmetic modulus to the FitIns message.
        """
        # Call the base class method to get the FitIns
        clients = super().configure_fit(server_round, parameters, client_manager)
        
        # Add the sign vector and the arithmetic modulus to the FitIns message
        for _, fit_ins in clients:
            fit_ins.config["arithmetic_modulus"] = arithmetic_modulus
            # fit_ins.config["sign_vector"] = sign_vector
        
        return clients