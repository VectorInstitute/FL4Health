import os
import warnings
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from flamby.datasets.fed_isic2019 import Baseline
from flwr.common.parameter import ndarrays_to_parameters
from flwr.common.typing import Config, Metrics, Parameters
from torch.utils.data import DataLoader

from examples.simple_metric_aggregation import metric_aggregation, normalize_metrics
from fl4health.utils.metrics import AccumulationMeter, Metric

warnings.filterwarnings("ignore", category=UserWarning)


def fit_config(
    local_steps: int,
    n_server_rounds: int,
    current_round: int,
) -> Config:
    return {
        "local_steps": local_steps,
        "n_server_rounds": n_server_rounds,
        "current_server_round": current_round,
    }


def get_initial_model_parameters(client_model: nn.Module) -> Parameters:
    # Initializing the model parameters on the server side.
    return ndarrays_to_parameters([val.cpu().numpy() for _, val in client_model.state_dict().items()])


def fit_metrics_aggregation_fn(all_client_metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # This function is run by the server to aggregate metrics returned by each clients fit function
    # NOTE: The first value of the tuple is number of examples for FedAvg
    total_examples, aggregated_metrics = metric_aggregation(all_client_metrics)
    return normalize_metrics(total_examples, aggregated_metrics)


def evaluate_metrics_aggregation_fn(all_client_metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # This function is run by the server to aggregate metrics returned by each clients evaluate function
    # NOTE: The first value of the tuple is number of examples for FedAvg
    total_examples, aggregated_metrics = metric_aggregation(all_client_metrics)
    return normalize_metrics(total_examples, aggregated_metrics)


def get_all_run_folders(artifact_dir: str) -> List[str]:
    run_folder_names = [folder_name for folder_name in os.listdir(artifact_dir) if "Run" in folder_name]
    return [os.path.join(artifact_dir, run_folder_name) for run_folder_name in run_folder_names]


def write_measurement_results(eval_write_path: str, results: Dict[str, float]) -> None:
    with open(eval_write_path, "w") as f:
        for key, metric_vaue in results.items():
            f.write(f"{key}: {metric_vaue}\n")


def load_local_model(run_folder_dir: str, client_number: int) -> Baseline:
    model_checkpoint_path = os.path.join(run_folder_dir, f"client_{client_number}_best_model.pkl")
    model = torch.load(model_checkpoint_path)
    return model


def load_global_model(run_folder_dir: str) -> Baseline:
    model_checkpoint_path = os.path.join(run_folder_dir, "server_best_model.pkl")
    model = torch.load(model_checkpoint_path)
    return model


def get_metric_avg_std(metrics: List[float]) -> Tuple[float, float]:
    mean = float(np.mean(metrics))
    std = float(np.std(metrics, ddof=1))
    return mean, std


def evaluate_fed_isic_model(
    model: nn.Module, dataset: DataLoader, metrics: Sequence[Metric], device: torch.device
) -> float:
    model.to(device).eval()
    meter = AccumulationMeter(metrics, "test_meter")

    with torch.no_grad():
        for input, target in dataset:
            input, target = input.to(device), target.to(device)
            preds = model(input)
            meter.update(preds, target)

    computed_metrics = meter.compute()
    assert "test_meter_FedIsic2019_balanced_accuracy" in computed_metrics
    balanced_accuracy = computed_metrics["test_meter_FedIsic2019_balanced_accuracy"]
    assert isinstance(balanced_accuracy, float)
    return balanced_accuracy
