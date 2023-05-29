import argparse
import os
from logging import INFO
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn
from flamby.datasets.fed_isic2019 import BATCH_SIZE, NUM_CLIENTS, Baseline, FedIsic2019, metric
from flwr.common.logger import log
from flwr.common.typing import Scalar, Tuple
from torch.utils.data import DataLoader

from fl4health.utils.metrics import AverageMeter, Metric


class FedIsic2019Metric(Metric):
    def __init__(self, name: str = "FedIsic2019_balanced_accuracy"):
        super().__init__(name)

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> Scalar:
        return metric(target, pred)


def get_all_run_folders(artifact_dir: str) -> List[str]:
    run_folder_names = [folder_name for folder_name in os.listdir(artifact_dir) if "Run" in folder_name]
    return [os.path.join(artifact_dir, run_folder_name) for run_folder_name in run_folder_names]


def write_measurement_results(results: Dict[str, float]) -> None:
    with open("test_processing_results.txt", "w") as f:
        for key, metric_vaue in results.items():
            f.write(f"{key}: {metric_vaue}\n")


def load_local_model(run_folder_dir: str, client_number: int) -> Baseline:
    model_checkpoint_path = os.path.join(run_folder_dir, f"client_{client_number}_best_model.pkl")
    model = torch.load(model_checkpoint_path)
    assert isinstance(model, Baseline)
    return model


def load_global_model(run_folder_dir: str) -> Baseline:
    model_checkpoint_path = os.path.join(run_folder_dir, "server_best_model.pkl")
    model = torch.load(model_checkpoint_path)
    assert isinstance(model, Baseline)
    return model


def get_metric_avg_std(metrics: List[float]) -> Tuple[float, float]:
    mean = np.mean(metrics)
    std = np.std(metrics, ddof=1)
    return mean, std


def evaluate_model(model: nn.Module, dataset: DataLoader, metrics: Sequence[Metric], device: torch.device) -> float:
    model.eval()
    meter = AverageMeter(metrics, "test_meter")

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


def main(artifact_dir: str, dataset_dir: str) -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    all_run_folder_dir = get_all_run_folders(artifact_dir)
    test_results: Dict[str, float] = {}
    metrics = [FedIsic2019Metric()]

    all_local_test_metrics = []

    # First we test each clients best model on local test data
    for client_number in range(NUM_CLIENTS):
        client_test_dataset = FedIsic2019(center=client_number, train=False, pooled=False, data_path=dataset_dir)
        test_loader = DataLoader(client_test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        test_metrics = []
        for run_folder_dir in all_run_folder_dir:
            model = load_local_model(run_folder_dir, client_number)
            run_metric = evaluate_model(model, test_loader, metrics, device)
            log(INFO, f"Client Number {client_number}, Run folder: {run_folder_dir}: Test Performance: {run_metric}")
            test_metrics.append(run_metric)
            all_local_test_metrics.append(run_metric)

        avg_test_metric, std_test_metric = get_metric_avg_std(test_metrics)
        log(INFO, f"Client Number {client_number} Average Test Performance: {avg_test_metric}")
        log(INFO, f"Client Number {client_number} St. Dev. Test Performance: {std_test_metric}")
        test_results[f"client_number_{client_number}_local_avg"] = avg_test_metric
        test_results[f"client_number_{client_number}_local_std"] = std_test_metric

    all_avg_test_metric, all_std_test_metric = get_metric_avg_std(all_local_test_metrics)
    test_results["all_local_test_metric_avg"] = all_avg_test_metric
    test_results["all_local_test_metric_std"] = all_std_test_metric

    # Next we test server checkpointed best model on pooled test data
    pooled_test_dataset = FedIsic2019(center=0, train=False, pooled=True)
    pooled_test_loader = DataLoader(pooled_test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    test_metrics = []
    for run_folder_dir in all_run_folder_dir:
        model = load_global_model(run_folder_dir)
        run_metric = evaluate_model(model, pooled_test_loader, metrics, device)
        log(INFO, f"Server, Run folder: {run_folder_dir}: Test Performance: {run_metric}")
        test_metrics.append(run_metric)

    avg_test_metric, std_test_metric = get_metric_avg_std(test_metrics)
    log(INFO, f"Server Average Test Performance: {avg_test_metric}")
    log(INFO, f"Server St. Dev. Test Performance: {std_test_metric}")
    test_results["server_avg"] = avg_test_metric
    test_results["server_std"] = std_test_metric

    write_measurement_results(test_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FL Server Main")
    parser.add_argument(
        "--artifact_dir",
        action="store",
        type=str,
        help="Path to save model artifacts to be evaluated",
        required=True,
    )
    parser.add_argument(
        "--dataset_dir",
        action="store",
        type=str,
        help="Path to the preprocessed FedIsic2019 Dataset (ex. path/to/fedisic2019)",
        required=True,
    )
    args = parser.parse_args()

    log(INFO, f"Artifact Directory: {args.artifact_dir}")
    log(INFO, f"Dataset Directory: {args.dataset_dir}")
    main(args.artifact_dir, args.dataset_dir)
