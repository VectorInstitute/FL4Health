import argparse
from logging import INFO
from pathlib import Path

import torch
from data.data import load_test_delirium, load_test_mortality
from evaluation.utils import (
    evaluate_model,
    get_all_run_folders,
    get_metric_avg_std,
    load_global_model,
    load_local_model,
    write_measurement_results,
)
from flwr.common.logger import log

from research.gemini.metrics.metrics import Accuracy, BinaryF1, BinaryRocAuc


def main(
    artifact_dir: str,
    dataset_dir: Path,
    eval_write_path: str,
    n_clients: int,
    hospitals: list[str],
    learning_task: str,
    eval_global_model: bool,
    is_apfl: bool,
) -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    all_run_folder_dir = get_all_run_folders(artifact_dir)

    metrics = [
        BinaryRocAuc("roc"),
        BinaryF1("f1"),
        Accuracy("accuracy"),
        # Binary_F1_Macro("macro_f1"),
        # Binary_Balanced_Accuracy("balanced_accuracy"),
    ]
    # metrics = []

    for metric in metrics:
        test_results: dict[str, float] = {}
        all_local_test_metrics = dict.fromkeys(all_run_folder_dir, 0.0)
        all_server_test_metrics = dict.fromkeys(all_run_folder_dir, 0.0)

        # First we test each client's best model on local test data and the best server model on that same data
        for client_number in range(n_clients):
            hospital_list = hospitals[client_number].split(" ")

            # load local test data
            if learning_task == "mortality":
                test_loader = load_test_mortality(data_path, 64, hospital_list)
            else:
                test_loader = load_test_delirium(data_path, 64, hospital_list)

            hospital_names = ",".join(hospital_list)
            print("hospital string", hospital_names)
            local_test_metrics = []
            server_test_metrics = []
            for run_folder_dir in all_run_folder_dir:
                local_model = load_local_model(run_folder_dir, hospital_names)
                local_run_metric = evaluate_model(local_model, test_loader, [metric], device, is_apfl)
                log(
                    INFO,
                    f"Client Number {hospital_names}, Run folder: {run_folder_dir}: "
                    f"Local Model Test Performance: {local_run_metric}",
                )
                local_test_metrics.append(local_run_metric)
                all_local_test_metrics[run_folder_dir] += local_run_metric / n_clients

                if eval_global_model:
                    server_model = load_global_model(run_folder_dir)
                    server_run_metric = evaluate_model(server_model, test_loader, [metric], device, is_apfl)
                    log(
                        INFO,
                        f"Client {hospital_names}, Run folder: {run_folder_dir}: "
                        f"Server Model Test Performance: {server_run_metric}",
                    )
                    server_test_metrics.append(server_run_metric)
                    all_server_test_metrics[run_folder_dir] += server_run_metric / n_clients
            avg_test_metric, std_test_metric = get_metric_avg_std(local_test_metrics)
            log(
                INFO,
                f"Client {hospital_names} Model Average Test Performance on own Data: {avg_test_metric}",
            )
            log(
                INFO,
                f"Client {hospital_names} Model St. Dev. Test Performance on own Data: {std_test_metric}",
            )
            test_results[f"client_{hospital_names}_model_local_avg"] = avg_test_metric
            test_results[f"client_{hospital_names}_model_local_std"] = std_test_metric

            if eval_global_model:
                (
                    avg_server_test_local_metric,
                    std_server_test_local_metric,
                ) = get_metric_avg_std(server_test_metrics)
                log(
                    INFO,
                    f"Server model Average Test Performance on Client {hospital_names} Data: {avg_test_metric}",
                )
                log(
                    INFO,
                    f"Server model St. Dev. Test Performance on Client {hospital_names} Data: {std_test_metric}",
                )
                test_results[f"server_model_client_{hospital_names}_avg"] = avg_server_test_local_metric
                test_results[f"server_model_client_{hospital_names}_std"] = std_server_test_local_metric

        all_avg_test_metric, all_std_test_metric = get_metric_avg_std(list(all_local_test_metrics.values()))
        test_results["avg_local_model_avg_across_clients"] = all_avg_test_metric
        test_results["std_local_model_avg_across_clients"] = all_std_test_metric
        log(
            INFO,
            f"Avg Local Model Test Performance Over all clients: {all_avg_test_metric}",
        )
        log(
            INFO,
            f"Std. Dev. Local Model Test Performance Over all clients: {all_std_test_metric}",
        )

        if eval_global_model:
            all_server_avg_test_metric, all_server_std_test_metric = get_metric_avg_std(
                list(all_server_test_metrics.values())
            )
            test_results["avg_server_model_avg_across_clients"] = all_server_avg_test_metric
            test_results["std_server_model_avg_across_clients"] = all_server_std_test_metric
            log(
                INFO,
                f"Avg. Server Model Test Performance Over all clients: {all_server_avg_test_metric}",
            )
            log(
                INFO,
                f"Std. Dev. Server Model Test Performance Over all clients: {all_server_std_test_metric}",
            )

        if eval_global_model:
            # Next we test server checkpointed best model on pooled test data
            if learning_task == "mortality":
                pooled_test_loader = load_test_mortality(data_path, 64)
            else:
                pooled_test_loader = load_test_delirium(data_path, 64)

            test_metrics = []
            for run_folder_dir in all_run_folder_dir:
                model = load_global_model(run_folder_dir)
                run_metric = evaluate_model(model, pooled_test_loader, [metric], device, is_apfl)
                log(
                    INFO,
                    f"Server, Run folder: {run_folder_dir}: Test Performance: {run_metric}",
                )
                test_metrics.append(run_metric)

            avg_server_test_metric, std_server_test_metric = get_metric_avg_std(test_metrics)
            log(
                INFO,
                f"Server Average Test Performance on Pooled Data: {avg_server_test_metric}",
            )
            log(
                INFO,
                f"Server St. Dev. Test Performance on Pooled Data: {std_server_test_metric}",
            )
            test_results["server_avg_pooled"] = avg_server_test_metric
            test_results["server_std_pooled"] = std_server_test_metric

        write_measurement_results(eval_write_path, metric, test_results)


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
        "--task",
        action="store",
        type=str,
        help="task: mortality",
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
    parser.add_argument(
        "--eval_global_model",
        action="store_true",
        help="boolean to indicate whether to search for and evaluate a server model in addition to client models",
    )
    parser.add_argument(
        "--is_apfl",
        action="store_true",
        help="boolean to indicate whether we are evaluating the apfl model or not",
    )

    args = parser.parse_args()
    if args.task == "mortality":
        data_path = Path("mortality_data")
        if args.n_clients == 2:
            hospitals = ["THPC THPM", "SMH MSH UHNTG UHNTW SBK"]
        elif args.n_clients == 7:
            hospitals = ["THPC", "THPM", "SMH", "MSH", "UHNTG", "UHNTW", "SBK"]
    elif args.task == "delirium":
        data_path = Path("heterogeneous_data")
        hospitals = ["100", "101", "103", "105", "106", "107"]

    log(INFO, f"Artifact Directory: {args.artifact_dir}")
    log(INFO, f"Eval Write Path: {args.eval_write_path}")
    log(INFO, f"Run Global Model: {args.eval_global_model}")
    log(INFO, f"Number of Clients: {args.n_clients}")
    main(
        args.artifact_dir,
        data_path,
        args.eval_write_path,
        args.n_clients,
        hospitals,
        args.task,
        args.eval_global_model,
        args.is_apfl,
    )
