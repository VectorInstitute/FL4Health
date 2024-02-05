import os
import warnings
from logging import INFO
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from flwr.common.logger import log
from flwr.common.parameter import ndarrays_to_parameters
from flwr.common.typing import Config, Metrics, Parameters
from torch.utils.data import DataLoader
from torchinfo import summary

from examples.simple_metric_aggregation import metric_aggregation, normalize_metrics
from fl4health.utils.metrics import Metric, MetricManager

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


def get_initial_model_info_with_control_variates(client_model: nn.Module) -> Tuple[Parameters, Parameters]:
    # Initializing the model parameters on the server side.
    model_weights = [val.cpu().numpy() for _, val in client_model.state_dict().items()]
    # Initializing the control variates to zero, as suggested in the originalq scaffold paper
    control_variates = [np.zeros_like(val.data) for val in client_model.parameters() if val.requires_grad]
    return ndarrays_to_parameters(model_weights), ndarrays_to_parameters(control_variates)


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


def load_local_model(run_folder_dir: str, client_number: int) -> nn.Module:
    model_checkpoint_path = os.path.join(run_folder_dir, f"client_{client_number}_best_model.pkl")
    model = torch.load(model_checkpoint_path)
    return model


def load_global_model(run_folder_dir: str) -> nn.Module:
    model_checkpoint_path = os.path.join(run_folder_dir, "server_best_model.pkl")
    model = torch.load(model_checkpoint_path)
    return model


def get_metric_avg_std(metrics: List[float]) -> Tuple[float, float]:
    mean = float(np.mean(metrics))
    std = float(np.std(metrics, ddof=1))
    return mean, std


def evaluate_model_on_dataset(
    model: nn.Module, dataset: DataLoader, metrics: Sequence[Metric], device: torch.device, is_apfl: bool
) -> MetricManager:
    model.to(device).eval()
    meter = MetricManager(metrics, "test_meter")

    with torch.no_grad():
        for input, target in dataset:
            input, target = input.to(device), target.to(device)
            if is_apfl:
                preds = model(input)["personal"]
            else:
                preds = model(input)
                if isinstance(preds, tuple):
                    preds = preds[0]
            preds = preds if isinstance(preds, dict) else {"prediction": preds}
            meter.update(preds, target)
    return meter


def evaluate_fed_isic_model(
    model: nn.Module, dataset: DataLoader, metrics: Sequence[Metric], device: torch.device, is_apfl: bool
) -> float:
    meter = evaluate_model_on_dataset(model, dataset, metrics, device, is_apfl)
    computed_metrics = meter.compute()
    assert "test_meter - prediction - FedIsic2019_balanced_accuracy" in computed_metrics
    balanced_accuracy = computed_metrics["test_meter - prediction - FedIsic2019_balanced_accuracy"]
    assert isinstance(balanced_accuracy, float)
    return balanced_accuracy


def evaluate_fed_heart_disease_model(
    model: nn.Module, dataset: DataLoader, metrics: Sequence[Metric], device: torch.device, is_apfl: bool
) -> float:
    meter = evaluate_model_on_dataset(model, dataset, metrics, device, is_apfl)

    computed_metrics = meter.compute()
    assert "test_meter - prediction - FedHeartDisease_accuracy" in computed_metrics
    accuracy = computed_metrics["test_meter - prediction - FedHeartDisease_accuracy"]
    assert isinstance(accuracy, float)
    return accuracy


def evaluate_fed_ixi_model(
    model: nn.Module, dataset: DataLoader, metrics: Sequence[Metric], device: torch.device, is_apfl: bool
) -> float:
    meter = evaluate_model_on_dataset(model, dataset, metrics, device, is_apfl)
    computed_metrics = meter.compute()
    assert "test_meter - prediction - FedIXI_dice" in computed_metrics
    dice = computed_metrics["test_meter - prediction - FedIXI_dice"]
    assert isinstance(dice, float)
    return dice


def summarize_model_info(model: nn.Module) -> None:
    model_stats = summary(model, verbose=0)
    log(INFO, "Model Stats:")
    log(INFO, "===========================================================================")
    log(INFO, f"Total Parameters: {model_stats.total_params}")
    log(INFO, f"Trainable Parameters: {model_stats.trainable_params}")
    log(INFO, f"Frozen Parameters: {model_stats.total_params - model_stats.trainable_params}")
    log(INFO, "===========================================================================\n")


def shutoff_batch_norm_tracking(model: nn.Module) -> None:
    # Iterate through all named modules of the model and, if we encounter a batch normalization layer, we set
    # track_running_stats to false instead of true.
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm3d) or isinstance(module, nn.BatchNorm2d):
            log(INFO, f"Modifying Batch Normalization Layer: {name}")
            module.track_running_stats = False
            # NOTE: It's apparently not enough to set this boolean to false. We need to set all of the relevant
            # variable to none, otherwise the layer still tries to apply the stale variables during evaluation
            # leading to eventual NaNs again.
            module.running_mean = None
            module.running_var = None
            module.num_batches_tracked = None
            module.register_buffer("running_mean", None)
            module.register_buffer("running_var", None)
            module.register_buffer("num_batches_tracked", None)
