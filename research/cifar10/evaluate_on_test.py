import argparse
from logging import INFO
from pathlib import Path
from typing import Dict

import torch
from flwr.common.logger import log

from fl4health.utils.load_data import load_cifar10_test_data
from fl4health.utils.metrics import Accuracy
from fl4health.utils.sampler import DirichletLabelBasedSampler
from research.cifar10.utils import (
    evaluate_cifar10_model,
    get_all_run_folders,
    get_metric_avg_std,
    load_global_model,
    load_local_model,
    write_measurement_results,
)

NUM_CLIENTS = 10
BATCH_SIZE = 32


def main(
    artifact_dir: str,
    dataset_dir: str,
    eval_write_path: str,
    eval_local_models: bool,
    eval_global_model: bool,
    heterogeneity_level: float,
    is_apfl: bool,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_run_folder_dir = get_all_run_folders(artifact_dir)
    test_results: Dict[str, float] = {}
    metrics = [Accuracy("cifar10_accuracy")]

    all_local_test_metrics = {run_folder_dir: 0.0 for run_folder_dir in all_run_folder_dir}
    all_server_test_metrics = {run_folder_dir: 0.0 for run_folder_dir in all_run_folder_dir}

    # First we test each client's best model on local test data and the best server model on that same data

    for client_number in range(NUM_CLIENTS):
        sampler = DirichletLabelBasedSampler(
            list(range(10)),
            sample_percentage=1.0 / NUM_CLIENTS,
            beta=heterogeneity_level,
            hash_key=client_number,
        )
        test_loader, _ = load_cifar10_test_data(Path(dataset_dir), BATCH_SIZE, sampler=sampler)

        local_test_metrics = []
        server_test_metrics = []
        for run_folder_dir in all_run_folder_dir:
            if eval_local_models:
                local_model = load_local_model(run_folder_dir, client_number)
                local_run_metric = evaluate_cifar10_model(local_model, test_loader, metrics, device, is_apfl)
                log(
                    INFO,
                    f"Client Number {client_number}, Run folder: {run_folder_dir}: "
                    f"Local Model Test Performance: {local_run_metric}",
                )
                local_test_metrics.append(local_run_metric)
                all_local_test_metrics[run_folder_dir] += local_run_metric / NUM_CLIENTS

            if eval_global_model:
                server_model = load_global_model(run_folder_dir)
                server_run_metric = evaluate_cifar10_model(server_model, test_loader, metrics, device, is_apfl)
                log(
                    INFO,
                    f"Client Number {client_number}, Run folder: {run_folder_dir}: "
                    f"Server Model Test Performance: {server_run_metric}",
                )
                server_test_metrics.append(server_run_metric)
                all_server_test_metrics[run_folder_dir] += server_run_metric / NUM_CLIENTS

        if eval_local_models:
            avg_test_metric, std_test_metric = get_metric_avg_std(local_test_metrics)
            log(INFO, f"Client {client_number} Model Average Test Performance on own Data: {avg_test_metric}")
            log(INFO, f"Client {client_number} Model St. Dev. Test Performance on own Data: {std_test_metric}")
            test_results[f"client_{client_number}_model_local_avg"] = avg_test_metric
            test_results[f"client_{client_number}_model_local_std"] = std_test_metric

        if eval_global_model:
            avg_server_test_local_metric, std_server_test_local_metric = get_metric_avg_std(server_test_metrics)
            log(
                INFO,
                f"Server model Average Test Performance on Client {client_number} "
                f"Data: {avg_server_test_local_metric}",
            )
            log(
                INFO,
                f"Server model St. Dev. Test Performance on Client {client_number} "
                f"Data: {std_server_test_local_metric}",
            )
            test_results[f"server_model_client_{client_number}_avg"] = avg_server_test_local_metric
            test_results[f"server_model_client_{client_number}_std"] = std_server_test_local_metric

    if eval_local_models:
        all_avg_test_metric, all_std_test_metric = get_metric_avg_std(list(all_local_test_metrics.values()))
        test_results["avg_local_model_avg_across_clients"] = all_avg_test_metric
        test_results["std_local_model_avg_across_clients"] = all_std_test_metric
        log(INFO, f"Avg Local Model Test Performance Over all clients: {all_avg_test_metric}")
        log(INFO, f"Std. Dev. Local Model Test Performance Over all clients: {all_std_test_metric}")

    if eval_global_model:
        all_server_avg_test_metric, all_server_std_test_metric = get_metric_avg_std(
            list(all_server_test_metrics.values())
        )
        test_results["avg_server_model_avg_across_clients"] = all_server_avg_test_metric
        test_results["std_server_model_avg_across_clients"] = all_server_std_test_metric
        log(INFO, f"Avg. Server Model Test Performance Over all clients: {all_server_avg_test_metric}")
        log(INFO, f"Std. Dev. Server Model Test Performance Over all clients: {all_server_std_test_metric}")

    write_measurement_results(eval_write_path, test_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Holdout Global")
    parser.add_argument(
        "--artifact_dir",
        action="store",
        type=str,
        help="Path to saved model artifacts to be evaluated",
        required=True,
    )
    parser.add_argument(
        "--dataset_dir",
        action="store",
        type=str,
        help="Path to the preprocessed Cifar10 Dataset (ex. path/to/cifar10)",
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
        "--eval_global_model",
        action="store_true",
        help="boolean to indicate whether to search for and evaluate a server model in addition to client models",
    )
    parser.add_argument(
        "--eval_local_models",
        action="store_true",
        help="boolean to indicate whether to search for and evaluate a local models in addition to the server model",
    )

    parser.add_argument(
        "--is_apfl",
        action="store_true",
        help="boolean to indicate whether we're evaluating an APFL model or not, as those model have special args",
    )
    parser.add_argument(
        "--beta",
        action="store",
        type=float,
        help="Heterogeneity level for the dataset",
        required=False,
        default=0.1,
    )

    args = parser.parse_args()
    log(INFO, f"Artifact Directory: {args.artifact_dir}")
    log(INFO, f"Dataset Directory: {args.dataset_dir}")
    log(INFO, f"Eval Write Path: {args.eval_write_path}")
    log(INFO, f"Run Local Models: {args.eval_local_models}")
    log(INFO, f"Run Global Model: {args.eval_global_model}")
    log(INFO, f"Heterogeneity level for the dataset: {args.beta}")
    log(INFO, f"Is APFL Run: {args.is_apfl}")

    assert args.eval_local_models or args.eval_global_model
    main(
        args.artifact_dir,
        args.dataset_dir,
        args.eval_write_path,
        args.eval_local_models,
        args.eval_global_model,
        args.beta,
        args.is_apfl,
    )