import argparse
import os
from logging import INFO
from pathlib import Path

import torch
from data.data import load_test_delirium, load_test_mortality
from evaluation.utils import evaluate_model, get_all_run_folders, get_metric_avg_std
from flwr.common.logger import log
from torch import nn

from research.gemini.metrics.metrics import Accuracy, BinaryF1, BinaryRocAuc


def load_centralized_model(run_folder_dir: str) -> nn.Module:
    model_checkpoint_path = os.path.join(run_folder_dir, "centralized_best_model.pkl")
    model = torch.load(model_checkpoint_path, weights_only=False)
    assert isinstance(model, nn.Module)
    return model


def write_measurement_results(eval_write_path: str, metric, results: dict[str, float]) -> None:
    metric_write_path = os.path.join(eval_write_path, f"{metric.name}_metric.txt")
    with open(metric_write_path, "w") as f:
        for key, metric_value in results.items():
            f.write(f"{key}: {metric_value}\n")


def main(
    data_path: Path, artifact_dir: str, eval_write_path: str, n_clients: int, hospitals: list[str], learning_task: str
) -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    all_run_folder_dir = get_all_run_folders(artifact_dir)

    metrics = [
        BinaryRocAuc("roc"),
        BinaryF1("f1"),
        Accuracy("accuracy"),
    ]

    if learning_task == "mortality":
        pooled_test_loader = load_test_mortality(data_path, 64)
    else:
        pooled_test_loader = load_test_delirium(data_path, 64)

    for metric in metrics:
        test_results: dict[str, float] = {}
        all_clients_test_metrics = dict.fromkeys(all_run_folder_dir, 0.0)
        client_test_metrics = {client_id: [] for client_id in range(n_clients)}

        pooled_metric_values = []
        for run_folder_dir in all_run_folder_dir:
            model = load_centralized_model(run_folder_dir)
            pooled_result = evaluate_model(model, pooled_test_loader, [metric], device, False)
            pooled_metric_values.append(pooled_result)

            # Take the average over all client's test data
            for client_num in range(n_clients):
                hospital_name = hospitals[client_num].split(" ")
                if learning_task == "mortality":
                    test_loader_client = load_test_mortality(data_path, 64, hospital_name)
                else:
                    test_loader_client = load_test_delirium(data_path, 64, hospital_name)
                clients_run_metric = evaluate_model(model, test_loader_client, [metric], device, False)

                all_clients_test_metrics[run_folder_dir] += clients_run_metric / n_clients
                client_test_metrics[client_num].append(clients_run_metric)

        mean, std = get_metric_avg_std(pooled_metric_values)
        test_results[f"Pooled_{metric.name}_avg"] = mean
        test_results[f"Pooled_{metric.name}_std"] = std

        all_avg_test_metric, all_std_test_metric = get_metric_avg_std(list(all_clients_test_metrics.values()))
        test_results["avg_central_model_avg_across_clients"] = all_avg_test_metric
        test_results["avg_central_model_std_across_clients"] = all_std_test_metric
        log(INFO, f"Avg Central Model Test Performance Over all clients: {all_avg_test_metric}")
        log(INFO, f"Std. Dev. Central Model Test Performance Over all clients: {all_std_test_metric}")

        for client_num in range(n_clients):
            hospital_name = hospitals[client_num].split(" ")
            client_avg_test_metric, client_std_test_metric = get_metric_avg_std(client_test_metrics[client_num])
            test_results[f"server_model_client_{hospital_name[0]}_avg"] = client_avg_test_metric
            test_results[f"server_model_client_{hospital_name[0]}_std"] = client_std_test_metric

        write_measurement_results(eval_write_path, metric, test_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Holdout centralized training")
    parser.add_argument(
        "--artifact_dir",
        action="store",
        type=str,
        help="Path to saved model artifacts to be evaluated",
        required=True,
    )
    parser.add_argument(
        "--task",
        action="store",
        type=str,
        help="task: mortality, or delirium",
        required=True,
    )
    parser.add_argument(
        "--eval_write_path",
        action="store",
        type=str,
        help="Path to write the evaluation results file",
        required=True,
    )
    parser.add_argument(
        "--n_clients",
        action="store",
        type=int,
        help="Number of the clients",
        required=True,
    )
    args = parser.parse_args()

    if args.task == "mortality":
        data_path = Path("mortality_data")
        if args.n_clients == 2:
            hospitals = ["THPC THPM", "SMH MSH UHNTG UHNTW SBK"]
        elif args.n_clients == 7:
            hospitals = ["THPC", "THPM", "SMH", "MSH", "UHNTG", "UHNTW", "SBK"]
    elif args.task == "delirium":
        data_path = Path("delirium_data")
        hospitals = ["100", "101", "103", "105", "106", "107"]

    main(data_path, args.artifact_dir, args.eval_write_path, args.n_clients, hospitals, args.task)
