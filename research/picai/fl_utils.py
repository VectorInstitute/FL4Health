from typing import Tuple, List
import torch.nn as nn
from flwr.common.typing import Parameters, Metrics, Config
from flwr.common.parameter import ndarrays_to_parameters
from research.picai.simple_metric_aggregation import normalize_metrics, metric_aggregation


def get_initial_model_parameters(client_model: nn.Module) -> Parameters:
    # Initializing the model parameters on the server side.
    return ndarrays_to_parameters([val.cpu().numpy() for _, val in client_model.state_dict().items()])


def fit_metrics_aggregation_fn(all_client_metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # This function is run by the server to aggregate metrics returned by each clients fit function
    # NOTE: The first value of the tuple is number of examples for FedAvg
    total_examples, aggregated_metrics = metric_aggregation(all_client_metrics)
    return normalize_metrics(total_examples, aggregated_metrics)


def fit_config(
    fold_id: int,
    batch_size: int,
    local_epochs: int,
    n_server_rounds: int,
    current_round: int,
) -> Config:
    return {
        "fold_id": fold_id,
        "batch_size": batch_size,
        "local_epochs": local_epochs,
        "n_server_rounds": n_server_rounds,
        "current_server_round": current_round,
    }


def evaluate_metrics_aggregation_fn(all_client_metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # This function is run by the server to aggregate metrics returned by each clients evaluate function
    # NOTE: The first value of the tuple is number of examples for FedAvg
    total_examples, aggregated_metrics = metric_aggregation(all_client_metrics)
    return normalize_metrics(total_examples, aggregated_metrics)