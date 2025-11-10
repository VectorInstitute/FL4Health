import os
from collections.abc import Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from fl4health.metrics.base_metrics import Metric
from fl4health.metrics.metric_managers import MetricManager


def get_all_run_folders(artifact_dir: str) -> list[str]:
    run_folder_names = [folder_name for folder_name in os.listdir(artifact_dir) if "Run" in folder_name]
    return [os.path.join(artifact_dir, run_folder_name) for run_folder_name in run_folder_names]


def load_best_global_model(run_folder_dir: str) -> nn.Module:
    model_checkpoint_path = os.path.join(run_folder_dir, "server_best_model.pkl")
    return torch.load(model_checkpoint_path, weights_only=False)


def load_last_global_model(run_folder_dir: str) -> nn.Module:
    model_checkpoint_path = os.path.join(run_folder_dir, "server_last_model.pkl")
    return torch.load(model_checkpoint_path, weights_only=False)


def get_metric_avg_std(metrics: list[float]) -> tuple[float, float]:
    mean = float(np.mean(metrics))
    std = float(np.std(metrics, ddof=1))
    return mean, std


def write_measurement_results(eval_write_path: str, results: dict[str, float]) -> None:
    with open(eval_write_path, "w") as f:
        for key, metric_value in results.items():
            f.write(f"{key}: {metric_value}\n")


def evaluate_synthetic_data_model(
    model: nn.Module, dataset: DataLoader, metrics: Sequence[Metric], device: torch.device, is_apfl: bool
) -> float:
    meter = evaluate_model_on_dataset(model, dataset, metrics, device, is_apfl)

    computed_metrics = meter.compute()
    assert "test_meter - prediction - synth_accuracy" in computed_metrics
    accuracy = computed_metrics["test_meter - prediction - synth_accuracy"]
    assert isinstance(accuracy, float)
    return accuracy


def load_eval_best_pre_aggregation_local_model(run_folder_dir: str, client_number: int) -> nn.Module:
    model_checkpoint_path = os.path.join(run_folder_dir, f"pre_aggregation_client_{client_number}_best_model.pkl")
    return torch.load(model_checkpoint_path, weights_only=False)


def load_eval_last_pre_aggregation_local_model(run_folder_dir: str, client_number: int) -> nn.Module:
    model_checkpoint_path = os.path.join(run_folder_dir, f"pre_aggregation_client_{client_number}_last_model.pkl")
    return torch.load(model_checkpoint_path, weights_only=False)


def load_eval_best_post_aggregation_local_model(run_folder_dir: str, client_number: int) -> nn.Module:
    model_checkpoint_path = os.path.join(run_folder_dir, f"post_aggregation_client_{client_number}_best_model.pkl")
    return torch.load(model_checkpoint_path, weights_only=False)


def load_eval_last_post_aggregation_local_model(run_folder_dir: str, client_number: int) -> nn.Module:
    model_checkpoint_path = os.path.join(run_folder_dir, f"post_aggregation_client_{client_number}_last_model.pkl")
    return torch.load(model_checkpoint_path, weights_only=False)


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
