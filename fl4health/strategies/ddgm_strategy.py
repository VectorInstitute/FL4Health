from logging import DEBUG, INFO, WARNING
from typing import Callable, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.linalg import vector_norm

from flwr.common import (
    parameters_to_ndarrays,
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

from fl4health.servers.secure_aggregation_utils import vectorize_model, unvectorize_model,get_model_layer_types, change_model_dtypes
from fl4health.utils.functions import decode_and_pseudo_sort_results

from fl4health.privacy_mechanisms.discrete_gaussian_mechanism import (
    fwht,
    shift_transform,
    shift_transform_torch
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
        beta: float = 0.9,
        ddgm_config: dict[str, Scalar] = {},
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
        self.global_lr = ddgm_config['global_lr']
        self.clipping_object = ddgm_config['clipping_object']

        assert self.clipping_object in ("diff_params", "params"), f"s must be 'diff_params' or 'params', but got '{self.clipping_object}'"

        self.beta = beta
        if self.beta != 0:
            log(INFO, f"enable server momentum update, momentum {self.beta}")
        self.m_t: torch.Tensor | None = None

    def calculate_update_with_momentum(self, weights_update: torch.Tensor) -> None:
        """
        Performs a weight update with momentum. That is, combining some weighted value of the previous update with
        the current update.

        Args:
            weights_update (Tensor, vectorized update): The current update after the weights have been aggregated from the training round.
        """
        if self.m_t is None:
            self.m_t = weights_update
        else:
            self.m_t = self.beta * self.m_t + (1 - self.beta) *  weights_update
        
    def aggregate_modular_sum(self, params: list[torch.Tensor], arithmetic_modulus: int) -> torch.Tensor:
        """
        Aggregate the parameters from the clients.
        """

        # Compute sum of weights of each layer
        summed = torch.stack(params).sum(dim=0)
        
        if (summed > arithmetic_modulus).any():
            # log(DEBUG, f"Modulus wrap up err included. modulus: {arithmetic_modulus}")
            log(DEBUG, f"Scale of summed params vector before modulus strategy: min {torch.min(summed)} max {torch.max(summed)} vector norm {vector_norm(summed)}")

        # Apply modular arithmetic
        aggregated = summed % arithmetic_modulus
        
        return aggregated
    
    def aggregated_sum(self, params: list[torch.Tensor]) -> torch.Tensor:
        return torch.stack(params).sum(dim=0)
    
    def ddgm_server_aggregation(self, client_vectors: list[torch.Tensor], arithmetic_modulus: int, sign_vector: torch.Tensor) -> torch.Tensor:
        """
        Aggregate the parameters from the clients.
        DDGM Server Process
        """
        num_clients = len(client_vectors)
        log(DEBUG, f"num of clients in this round to avg: {num_clients}")
        
        # aggregate (sum) the results
        aggregated_vector = self.aggregate_modular_sum(client_vectors, arithmetic_modulus) 

        log(DEBUG, f"Scale of aggregated params vector before avg strategy: min {torch.min(aggregated_vector)} max {torch.max(aggregated_vector)} vector norm {vector_norm(aggregated_vector)}")
        
        # server procedure
        vector = shift_transform_torch(aggregated_vector, arithmetic_modulus)

        log(DEBUG, f"Scale of aggregated params vector after shifting: min {torch.min(vector)} max {torch.max(vector)} vector norm {vector_norm(vector)}")
        
        vector = fwht(vector).to(device)
        vector = self.granularity * sign_vector.to(device) * vector.to(device)
        
        vector = vector[:self.model_dim] 
        vector /= num_clients

        log(DEBUG, f"Scale of aggregated params vector after avg strategy: min {torch.min(vector)} max {torch.max(vector)} vector norm {vector_norm(vector)}")

        log(INFO, f'finished server procedure')
        return vector
        
    
    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
        arithmetic_modulus: int,
        sign_vector: torch.Tensor,
        server_model: nn.Module
    ) -> tuple[Parameters | None, dict[str, Scalar]]:
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}
        
        #  process the results
        # each element in deltas is a noisy vector from the client
        client_vectors = [torch.from_numpy(parameters_to_ndarrays(client_response.parameters)[0]) for _, client_response in results]

        log(INFO, f'type of element in results: {type(client_vectors[0])}')
        
        # apply the DDGM server aggregation
        aggregated_vector = self.ddgm_server_aggregation(client_vectors, arithmetic_modulus, sign_vector)
        log(DEBUG, f"Scale of aggregated secure vector: min {torch.min(aggregated_vector)} max {torch.max(aggregated_vector)} vector norm {vector_norm(aggregated_vector)}")

        dtypes = get_model_layer_types(server_model)

        # WARNING: for debug use only
        client_ground_truth_vectors = [torch.from_numpy(parameters_to_ndarrays(client_response.parameters)[1]) for _, client_response in results]
        aggregated_ground_truth = self.aggregated_sum(client_ground_truth_vectors).to(device=aggregated_vector.device)
        aggregated_ground_truth /= len(client_ground_truth_vectors)
        log(DEBUG, f'Scale of aggregated_sum ground truth: min {torch.min(aggregated_ground_truth)} max {torch.max(aggregated_ground_truth)} vector norm {vector_norm(aggregated_ground_truth)}')
        gap_gt_sec = aggregated_vector - aggregated_ground_truth
        log(DEBUG, f"Scale of gap between gt and sec vector: min {torch.min(gap_gt_sec)} max {torch.max(gap_gt_sec)} vector norm {vector_norm(gap_gt_sec)}")

        if self.beta != 0:
            # enable momentum at server side
            self.calculate_update_with_momentum(aggregated_vector)
            aggregated_vector = self.m_t

        if self.clipping_object == "diff_params":
            
            aggregated_vector = vectorize_model(server_model) - aggregated_vector * self.global_lr
        
        server_model = unvectorize_model(server_model, aggregated_vector)

        self.server_model = change_model_dtypes(server_model, dtypes)

        self.parameters = ndarrays_to_parameters(
            [layer.cpu().numpy() for layer in self.server_model.state_dict().values()]
        )
        
        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return self.parameters, metrics_aggregated
    
    def configure_fit(
        self, 
        server_round: int, 
        parameters: Parameters, 
        client_manager: ClientManager,
        arithmetic_modulus: int,
    ) -> list[tuple[ClientProxy, FitIns]]:
        """
        Add the sign vector and the arithmetic modulus to the FitIns message.
        """
        # Call the base class method to get the FitIns
        clients = super().configure_fit(server_round, parameters, client_manager)
        
        # Add the sign vector and the arithmetic modulus to the FitIns message
        for _, fit_ins in clients:
            fit_ins.config["arithmetic_modulus"] = arithmetic_modulus
        
        return clients