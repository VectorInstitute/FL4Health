import os
import warnings
from collections.abc import Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from fl4health.metrics.base_metrics import Metric
from fl4health.metrics.metric_managers import MetricManager


warnings.filterwarnings("ignore", category=UserWarning)


def get_all_run_folders(artifact_dir: str) -> list[str]:
    run_folder_names = [folder_name for folder_name in os.listdir(artifact_dir) if "Run" in folder_name]
    return [os.path.join(artifact_dir, run_folder_name) for run_folder_name in run_folder_names]


def write_measurement_results(eval_write_path: str, metric, results: dict[str, float]) -> None:
    metric_write_path = os.path.join(eval_write_path, f"{metric.name}_metric.txt")
    with open(metric_write_path, "w") as f:
        for key, metric_value in results.items():
            f.write(f"{key}: {metric_value}\n")


def load_local_model(run_folder_dir: str, hospital_names: str) -> nn.Module:
    model_checkpoint_path = os.path.join(run_folder_dir, f"client_{hospital_names}_best_model.pkl")
    model = torch.load(model_checkpoint_path, weights_only=False)
    assert isinstance(model, nn.Module)
    return model


def load_global_model(run_folder_dir: str) -> nn.Module:
    model_checkpoint_path = os.path.join(run_folder_dir, "server_best_model.pkl")
    model = torch.load(model_checkpoint_path, weights_only=False)
    assert isinstance(model, nn.Module)
    return model


def get_metric_avg_std(metrics: list[float]) -> tuple[float, float]:
    mean = float(np.mean(metrics))
    std = float(np.std(metrics, ddof=1))
    return mean, std


def evaluate_model(
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
                print(preds)
            preds = preds if isinstance(preds, dict) else {"prediction": preds}
            meter.update(preds, target)
    computed_metric = meter.compute()
    print(computed_metric)
    return computed_metric[f"test_meter - prediction - {metrics[0].name}"]
