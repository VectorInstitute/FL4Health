import argparse
import copy
from logging import INFO
from pathlib import Path

import torch
from flwr.common.logger import log

from fl4health.datasets.rxrx1.load_data import load_rxrx1_test_data
from fl4health.metrics import Accuracy
from fl4health.utils.dataset import TensorDataset
from research.rxrx1.utils import (
    evaluate_rxrx1_model,
    get_all_run_folders,
    get_metric_avg_std,
    load_best_global_model,
    load_eval_best_post_aggregation_local_model,
    load_eval_best_pre_aggregation_local_model,
    load_eval_last_post_aggregation_local_model,
    load_eval_last_pre_aggregation_local_model,
    load_last_global_model,
    write_measurement_results,
)


NUM_CLIENTS = 4
BATCH_SIZE = 32


def main(
    artifact_dir: str,
    dataset_dir: str,
    eval_write_path: str,
    eval_best_pre_aggregation_local_models: bool,
    eval_last_pre_aggregation_local_models: bool,
    eval_best_post_aggregation_local_models: bool,
    eval_last_post_aggregation_local_models: bool,
    eval_best_global_model: bool,
    eval_last_global_model: bool,
    eval_over_aggregated_test_data: bool,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_run_folder_dir = get_all_run_folders(artifact_dir)
    test_results: dict[str, float] = {}
    metrics = [Accuracy("rxrx1_accuracy")]

    all_pre_best_local_test_metrics = dict.fromkeys(all_run_folder_dir, 0.0)
    all_pre_last_local_test_metrics = dict.fromkeys(all_run_folder_dir, 0.0)
    all_post_best_local_test_metrics = dict.fromkeys(all_run_folder_dir, 0.0)
    all_post_last_local_test_metrics = dict.fromkeys(all_run_folder_dir, 0.0)

    all_best_server_test_metrics = dict.fromkeys(all_run_folder_dir, 0.0)
    all_last_server_test_metrics = dict.fromkeys(all_run_folder_dir, 0.0)

    if eval_over_aggregated_test_data:
        all_pre_best_local_agg_test_metrics = dict.fromkeys(all_run_folder_dir, 0.0)
        all_pre_last_local_agg_test_metrics = dict.fromkeys(all_run_folder_dir, 0.0)
        all_post_best_local_agg_test_metrics = dict.fromkeys(all_run_folder_dir, 0.0)
        all_post_last_local_agg_test_metrics = dict.fromkeys(all_run_folder_dir, 0.0)

        all_best_server_agg_test_metrics = dict.fromkeys(all_run_folder_dir, 0.0)
        all_last_server_agg_test_metrics = dict.fromkeys(all_run_folder_dir, 0.0)

    if eval_over_aggregated_test_data:
        for client_number in range(NUM_CLIENTS):
            test_loader, num_examples = load_rxrx1_test_data(
                data_path=Path(dataset_dir), client_num=client_number, batch_size=BATCH_SIZE
            )
            assert isinstance(test_loader.dataset, TensorDataset), "Expected TensorDataset."

            if client_number == 0:
                aggregated_dataset = copy.deepcopy(test_loader.dataset)
            else:
                assert aggregated_dataset.data is not None and test_loader.dataset.data is not None
                aggregated_dataset.data = torch.cat((aggregated_dataset.data, test_loader.dataset.data))
                assert aggregated_dataset.targets is not None and test_loader.dataset.targets is not None
                aggregated_dataset.targets = torch.cat((aggregated_dataset.targets, test_loader.dataset.targets))

        aggregated_test_loader = torch.utils.data.DataLoader(aggregated_dataset, batch_size=BATCH_SIZE, shuffle=False)
        aggregated_num_examples = len(aggregated_dataset)

    for client_number in range(NUM_CLIENTS):
        test_loader, num_examples = load_rxrx1_test_data(
            data_path=Path(dataset_dir), client_num=client_number, batch_size=BATCH_SIZE
        )

        pre_best_local_test_metrics = []
        pre_last_local_test_metrics = []
        post_best_local_test_metrics = []
        post_last_local_test_metrics = []
        best_server_test_metrics = []
        last_server_test_metrics = []

        if eval_over_aggregated_test_data:
            pre_best_local_agg_test_metrics = []
            pre_last_local_agg_test_metrics = []
            post_best_local_agg_test_metrics = []
            post_last_local_agg_test_metrics = []
            best_server_agg_test_metrics = []
            last_server_agg_test_metrics = []

        for run_folder_dir in all_run_folder_dir:
            if eval_best_pre_aggregation_local_models:
                local_model = load_eval_best_pre_aggregation_local_model(run_folder_dir, client_number)
                local_run_metric = evaluate_rxrx1_model(local_model, test_loader, metrics, device)
                log(
                    INFO,
                    f"Client Number {client_number}, Run folder: {run_folder_dir}: "
                    f"Best Pre-aggregation Local Model Test Performance: {local_run_metric}",
                )
                pre_best_local_test_metrics.append(local_run_metric)
                # Perform weighted average of the local model performance across all clients based on the number
                # of examples in the evaluation set
                all_pre_best_local_test_metrics[run_folder_dir] += (
                    local_run_metric * num_examples["eval_set"] / aggregated_num_examples
                )
                if eval_over_aggregated_test_data:
                    agg_local_run_metric = evaluate_rxrx1_model(local_model, aggregated_test_loader, metrics, device)
                    log(
                        INFO,
                        f"Client Number {client_number}, Run folder: {run_folder_dir}: "
                        f"Best Pre-aggregation Local Model Aggregated Test Performance: {agg_local_run_metric}",
                    )
                    pre_best_local_agg_test_metrics.append(agg_local_run_metric)
                    all_pre_best_local_agg_test_metrics[run_folder_dir] += agg_local_run_metric / NUM_CLIENTS

            if eval_last_pre_aggregation_local_models:
                local_model = load_eval_last_pre_aggregation_local_model(run_folder_dir, client_number)
                local_run_metric = evaluate_rxrx1_model(local_model, test_loader, metrics, device)
                log(
                    INFO,
                    f"Client Number {client_number}, Run folder: {run_folder_dir}: "
                    f"Last Pre-aggregation Local Model Test Performance: {local_run_metric}",
                )
                pre_last_local_test_metrics.append(local_run_metric)
                # Perform weighted average of the local model performance across all clients based on the number
                # of examples in the evaluation set
                all_pre_last_local_test_metrics[run_folder_dir] += (
                    local_run_metric * num_examples["eval_set"] / aggregated_num_examples
                )

                if eval_over_aggregated_test_data:
                    agg_local_run_metric = evaluate_rxrx1_model(local_model, aggregated_test_loader, metrics, device)
                    log(
                        INFO,
                        f"Client Number {client_number}, Run folder: {run_folder_dir}: "
                        f"Last Pre-aggregation Local Model Aggregated Test Performance: {agg_local_run_metric}",
                    )
                    pre_last_local_agg_test_metrics.append(agg_local_run_metric)
                    all_pre_last_local_agg_test_metrics[run_folder_dir] += agg_local_run_metric / NUM_CLIENTS

            if eval_best_post_aggregation_local_models:
                local_model = load_eval_best_post_aggregation_local_model(run_folder_dir, client_number)
                local_run_metric = evaluate_rxrx1_model(local_model, test_loader, metrics, device)
                log(
                    INFO,
                    f"Client Number {client_number}, Run folder: {run_folder_dir}: "
                    f"Best Post-aggregation Local Model Test Performance: {local_run_metric}",
                )
                post_best_local_test_metrics.append(local_run_metric)
                # Perform weighted average of the local model performance across all clients based on the number
                # of examples in the evaluation set
                all_post_best_local_test_metrics[run_folder_dir] += (
                    local_run_metric * num_examples["eval_set"] / aggregated_num_examples
                )

                if eval_over_aggregated_test_data:
                    agg_local_run_metric = evaluate_rxrx1_model(local_model, aggregated_test_loader, metrics, device)
                    log(
                        INFO,
                        f"Client Number {client_number}, Run folder: {run_folder_dir}: "
                        f"Best Post-aggregation Local Model Aggregated Test Performance: {agg_local_run_metric}",
                    )
                    post_best_local_agg_test_metrics.append(agg_local_run_metric)
                    all_post_best_local_agg_test_metrics[run_folder_dir] += agg_local_run_metric / NUM_CLIENTS

            if eval_last_post_aggregation_local_models:
                local_model = load_eval_last_post_aggregation_local_model(run_folder_dir, client_number)
                local_run_metric = evaluate_rxrx1_model(local_model, test_loader, metrics, device)
                log(
                    INFO,
                    f"Client Number {client_number}, Run folder: {run_folder_dir}: "
                    f"Last Post-aggregation Local Model Test Performance: {local_run_metric}",
                )
                post_last_local_test_metrics.append(local_run_metric)
                # Perform weighted average of the local model performance across all clients based on the number
                # of examples in the evaluation set
                all_post_last_local_test_metrics[run_folder_dir] += (
                    local_run_metric * num_examples["eval_set"] / aggregated_num_examples
                )

                if eval_over_aggregated_test_data:
                    agg_local_run_metric = evaluate_rxrx1_model(local_model, aggregated_test_loader, metrics, device)
                    log(
                        INFO,
                        f"Client Number {client_number}, Run folder: {run_folder_dir}: "
                        f"Last Post-aggregation Local Model Aggregated Test Performance: {agg_local_run_metric}",
                    )
                    post_last_local_agg_test_metrics.append(agg_local_run_metric)
                    all_post_last_local_agg_test_metrics[run_folder_dir] += agg_local_run_metric / NUM_CLIENTS

            if eval_best_global_model:
                server_model = load_best_global_model(run_folder_dir)
                server_run_metric = evaluate_rxrx1_model(server_model, test_loader, metrics, device)
                log(
                    INFO,
                    f"Client Number {client_number}, Run folder: {run_folder_dir}: "
                    f"Server Best Model Test Performance: {server_run_metric}",
                )
                best_server_test_metrics.append(server_run_metric)
                # Perform weighted average of the server model performance across all clients based on the number
                # of examples in the evaluation set
                all_best_server_test_metrics[run_folder_dir] += (
                    server_run_metric * num_examples["eval_set"] / aggregated_num_examples
                )

                if eval_over_aggregated_test_data:
                    agg_server_run_metric = evaluate_rxrx1_model(server_model, aggregated_test_loader, metrics, device)
                    log(
                        INFO,
                        f"Client Number {client_number}, Run folder: {run_folder_dir}: "
                        f"Server Best Model Aggregated Test Performance: {agg_server_run_metric}",
                    )
                    best_server_agg_test_metrics.append(agg_server_run_metric)
                    all_best_server_agg_test_metrics[run_folder_dir] += agg_server_run_metric / NUM_CLIENTS

            if eval_last_global_model:
                server_model = load_last_global_model(run_folder_dir)
                server_run_metric = evaluate_rxrx1_model(server_model, test_loader, metrics, device)
                log(
                    INFO,
                    f"Client Number {client_number}, Run folder: {run_folder_dir}: "
                    f"Server Last Model Test Performance: {server_run_metric}",
                )
                last_server_test_metrics.append(server_run_metric)
                # Perform weighted average of the server model performance across all clients based on the number
                # of examples in the evaluation set
                all_last_server_test_metrics[run_folder_dir] += (
                    server_run_metric * num_examples["eval_set"] / aggregated_num_examples
                )

                if eval_over_aggregated_test_data:
                    agg_server_run_metric = evaluate_rxrx1_model(server_model, aggregated_test_loader, metrics, device)
                    log(
                        INFO,
                        f"Client Number {client_number}, Run folder: {run_folder_dir}: "
                        f"Server Last Model Aggregated Test Performance: {agg_server_run_metric}",
                    )
                    last_server_agg_test_metrics.append(agg_server_run_metric)
                    all_last_server_agg_test_metrics[run_folder_dir] += agg_server_run_metric / NUM_CLIENTS

        # Write the results for each client
        if eval_best_pre_aggregation_local_models:
            avg_test_metric, std_test_metric = get_metric_avg_std(pre_best_local_test_metrics)
            log(
                INFO,
                f"""Client {client_number} Pre-aggregation Best Model Average Test
                    Performance on own Data: {avg_test_metric}""",
            )
            log(
                INFO,
                f"""Client {client_number} Pre-aggregation Best Model St. Dev. Test
                  Performance on own Data: {std_test_metric}""",
            )
            test_results[f"client_{client_number}_pre_best_model_local_avg"] = avg_test_metric
            test_results[f"client_{client_number}_pre_best_model_local_std"] = std_test_metric

            if eval_over_aggregated_test_data:
                avg_test_metric, std_test_metric = get_metric_avg_std(pre_best_local_agg_test_metrics)
                log(
                    INFO,
                    f"""Client {client_number} Pre-aggregation Best Model Average Aggregated Test
                        Performance: {avg_test_metric}""",
                )
                log(
                    INFO,
                    f"""Client {client_number} Pre-aggregation Best Model St. Dev. Aggregated Test
                    Performance: {std_test_metric}""",
                )
                test_results[f"agg_client_{client_number}_pre_best_model_local_avg"] = avg_test_metric
                test_results[f"agg_client_{client_number}_pre_best_model_local_std"] = std_test_metric

        if eval_last_pre_aggregation_local_models:
            avg_test_metric, std_test_metric = get_metric_avg_std(pre_last_local_test_metrics)
            log(
                INFO,
                f"""Client {client_number} Pre-aggregation Last Model Average Test
                Performance on own Data: {avg_test_metric}""",
            )
            log(
                INFO,
                f"""Client {client_number} Pre-aggregation Last Model St. Dev. Test
                  Performance on own Data: {std_test_metric}""",
            )
            test_results[f"client_{client_number}_pre_last_model_local_avg"] = avg_test_metric
            test_results[f"client_{client_number}_pre_last_model_local_std"] = std_test_metric
            if eval_over_aggregated_test_data:
                avg_test_metric, std_test_metric = get_metric_avg_std(pre_last_local_agg_test_metrics)
                log(
                    INFO,
                    f"""Client {client_number} Pre-aggregation Last Model Average Aggregated Test
                    Performance: {avg_test_metric}""",
                )
                log(
                    INFO,
                    f"""Client {client_number} Pre-aggregation Last Model St. Dev. Aggregated Test
                    Performance: {std_test_metric}""",
                )
                test_results[f"agg_client_{client_number}_pre_last_model_local_avg"] = avg_test_metric
                test_results[f"agg_client_{client_number}_pre_last_model_local_std"] = std_test_metric

        if eval_best_post_aggregation_local_models:
            avg_test_metric, std_test_metric = get_metric_avg_std(post_best_local_test_metrics)
            log(
                INFO,
                f"""Client {client_number} Post-aggregation Best Model Average Test
                  Performance on own Data: {avg_test_metric}""",
            )
            log(
                INFO,
                f"""Client {client_number} Post-aggregation Best Model St. Dev. Test
                  Performance on own Data: {std_test_metric}""",
            )
            test_results[f"client_{client_number}_post_best_model_local_avg"] = avg_test_metric
            test_results[f"client_{client_number}_post_best_model_local_std"] = std_test_metric

            if eval_over_aggregated_test_data:
                avg_test_metric, std_test_metric = get_metric_avg_std(post_best_local_agg_test_metrics)
                log(
                    INFO,
                    f"""Client {client_number} Post-aggregation Best Model Average Aggregated Test
                    Performance: {avg_test_metric}""",
                )
                log(
                    INFO,
                    f"""Client {client_number} Post-aggregation Best Model St. Dev. Aggregated Test
                    Performance: {std_test_metric}""",
                )
                test_results[f"agg_client_{client_number}_post_best_model_local_avg"] = avg_test_metric
                test_results[f"agg_client_{client_number}_post_best_model_local_std"] = std_test_metric

        if eval_last_post_aggregation_local_models:
            avg_test_metric, std_test_metric = get_metric_avg_std(post_last_local_test_metrics)
            log(
                INFO,
                f"""Client {client_number} Post-aggregation Last Model Average Test
                  Performance on own Data: {avg_test_metric}""",
            )
            log(
                INFO,
                f"""Client {client_number} Post-aggregation Last Model St. Dev. Test
                  Performance on own Data: {std_test_metric}""",
            )
            test_results[f"client_{client_number}_post_last_model_local_avg"] = avg_test_metric
            test_results[f"client_{client_number}_post_last_model_local_std"] = std_test_metric

            if eval_over_aggregated_test_data:
                avg_test_metric, std_test_metric = get_metric_avg_std(post_last_local_agg_test_metrics)
                log(
                    INFO,
                    f"""Client {client_number} Post-aggregation Last Model Average Aggregated Test
                    Performance: {avg_test_metric}""",
                )
                log(
                    INFO,
                    f"""Client {client_number} Post-aggregation Last Model St. Dev. Aggregated Test
                    Performance: {std_test_metric}""",
                )
                test_results[f"agg_client_{client_number}_post_last_model_local_avg"] = avg_test_metric
                test_results[f"agg_client_{client_number}_post_last_model_local_std"] = std_test_metric

        if eval_best_global_model:
            avg_server_test_global_metric, std_server_test_global_metric = get_metric_avg_std(best_server_test_metrics)
            log(
                INFO,
                f"Server Best model Average Test Performance on Client {client_number} "
                f"Data: {avg_server_test_global_metric}",
            )
            log(
                INFO,
                f"Server Best model St. Dev. Test Performance on Client {client_number} "
                f"Data: {std_server_test_global_metric}",
            )
            test_results[f"server_best_model_client_{client_number}_avg"] = avg_server_test_global_metric
            test_results[f"server_best_model_client_{client_number}_std"] = std_server_test_global_metric

        if eval_last_global_model:
            avg_server_test_global_metric, std_server_test_global_metric = get_metric_avg_std(last_server_test_metrics)
            log(
                INFO,
                f"Server Last model Average Test Performance on Client {client_number} "
                f"Data: {avg_server_test_global_metric}",
            )
            log(
                INFO,
                f"Server Last model St. Dev. Test Performance on Client {client_number} "
                f"Data: {std_server_test_global_metric}",
            )
            test_results[f"server_last_model_client_{client_number}_avg"] = avg_server_test_global_metric
            test_results[f"server_last_model_client_{client_number}_std"] = std_server_test_global_metric

    if eval_over_aggregated_test_data:
        if eval_best_global_model:
            avg_server_test_global_metric, std_server_test_global_metric = get_metric_avg_std(
                best_server_agg_test_metrics
            )
            log(
                INFO,
                f"Server Best model Average Test Performance on Aggregated Client Data"
                f"Data: {avg_server_test_global_metric}",
            )
            log(
                INFO,
                f"Server Best model St. Dev. Test Performance on Aggregated Client Data"
                f"Data: {std_server_test_global_metric}",
            )
            test_results["agg_server_best_model_client_avg"] = avg_server_test_global_metric
            test_results["agg_server_best_model_client_std"] = std_server_test_global_metric

        if eval_last_global_model:
            avg_server_test_global_metric, std_server_test_global_metric = get_metric_avg_std(
                last_server_agg_test_metrics
            )
            log(
                INFO,
                f"Server Last model Average Test Performance on Aggregated Client Data"
                f"Data: {avg_server_test_global_metric}",
            )
            log(
                INFO,
                f"Server Last model St. Dev. Test Performance on Aggregated Client Data"
                f"Data: {std_server_test_global_metric}",
            )
            test_results["agg_server_last_model_client_avg"] = avg_server_test_global_metric
            test_results["agg_server_last_model_client_std"] = std_server_test_global_metric

    if eval_best_pre_aggregation_local_models:
        all_avg_test_metric, all_std_test_metric = get_metric_avg_std(list(all_pre_best_local_test_metrics.values()))
        test_results["avg_pre_best_local_model_avg_across_clients"] = all_avg_test_metric
        test_results["std_pre_best_local_model_avg_across_clients"] = all_std_test_metric
        log(INFO, f"Avg Pre-aggregation Best Local Model Test Performance Over all clients: {all_avg_test_metric}")
        log(
            INFO,
            f"Std. Dev. Pre-aggregation Best Local Model Test Performance Over all clients: {all_std_test_metric}",
        )
        if eval_over_aggregated_test_data:
            all_avg_test_metric, all_std_test_metric = get_metric_avg_std(
                list(all_pre_best_local_agg_test_metrics.values())
            )
            test_results["agg_avg_pre_best_local_model_avg_across_clients"] = all_avg_test_metric
            test_results["agg_std_pre_best_local_model_avg_across_clients"] = all_std_test_metric
            log(
                INFO,
                f"""Avg Pre-aggregation Best Local Model Test
                  Performance Over Aggregated clients: {all_avg_test_metric}""",
            )
            log(
                INFO,
                f"""Std. Dev. Pre-aggregation Best Local Model Test
                  Performance Over Aggregated clients: {all_std_test_metric}""",
            )

    if eval_last_pre_aggregation_local_models:
        all_avg_test_metric, all_std_test_metric = get_metric_avg_std(list(all_pre_last_local_test_metrics.values()))
        test_results["avg_pre_last_local_model_avg_across_clients"] = all_avg_test_metric
        test_results["std_pre_last_local_model_avg_across_clients"] = all_std_test_metric
        log(INFO, f"Avg Pre-aggregation Last Local Model Test Performance Over all clients: {all_avg_test_metric}")
        log(
            INFO,
            f"Std. Dev. Pre-aggregation Last Local Model Test Performance Over all clients: {all_std_test_metric}",
        )
        if eval_over_aggregated_test_data:
            all_avg_test_metric, all_std_test_metric = get_metric_avg_std(
                list(all_pre_last_local_agg_test_metrics.values())
            )
            test_results["agg_avg_pre_last_local_model_avg_across_clients"] = all_avg_test_metric
            test_results["agg_std_pre_last_local_model_avg_across_clients"] = all_std_test_metric
            log(
                INFO,
                f"""Avg Pre-aggregation Last Local Model Test
                  Performance Over Aggregated clients: {all_avg_test_metric}""",
            )
            log(
                INFO,
                f"""Std. Dev. Pre-aggregation Last Local Model Test
                  Performance Over Aggregated clients: {all_std_test_metric}""",
            )

    if eval_best_post_aggregation_local_models:
        all_avg_test_metric, all_std_test_metric = get_metric_avg_std(list(all_post_best_local_test_metrics.values()))
        test_results["avg_post_best_local_model_avg_across_clients"] = all_avg_test_metric
        test_results["std_post_best_local_model_avg_across_clients"] = all_std_test_metric
        log(INFO, f"Avg Post-aggregation Best Local Model Test Performance Over all clients: {all_avg_test_metric}")
        log(
            INFO,
            f"Std. Dev. Post-aggregation Best Local Model Test Performance Over all clients: {all_std_test_metric}",
        )
        if eval_over_aggregated_test_data:
            all_avg_test_metric, all_std_test_metric = get_metric_avg_std(
                list(all_post_best_local_agg_test_metrics.values())
            )
            test_results["agg_avg_post_best_local_model_avg_across_clients"] = all_avg_test_metric
            test_results["agg_std_post_best_local_model_avg_across_clients"] = all_std_test_metric
            log(
                INFO,
                f"""Avg Post-aggregation Best Local Model Test
                  Performance Over Aggregated clients: {all_avg_test_metric}""",
            )
            log(
                INFO,
                f"""Std. Dev. Post-aggregation Best Local Model Test
                  Performance Over Aggregated clients: {all_std_test_metric}""",
            )

    if eval_last_post_aggregation_local_models:
        all_avg_test_metric, all_std_test_metric = get_metric_avg_std(list(all_post_last_local_test_metrics.values()))
        test_results["avg_post_last_local_model_avg_across_clients"] = all_avg_test_metric
        test_results["std_post_last_local_model_avg_across_clients"] = all_std_test_metric
        log(INFO, f"Avg Post-aggregation Last Local Model Test Performance Over all clients: {all_avg_test_metric}")
        log(
            INFO,
            f"Std. Dev. Post-aggregation Last Local Model Test Performance Over all clients: {all_std_test_metric}",
        )
        if eval_over_aggregated_test_data:
            all_avg_test_metric, all_std_test_metric = get_metric_avg_std(
                list(all_post_last_local_agg_test_metrics.values())
            )
            test_results["agg_avg_post_last_local_model_avg_across_clients"] = all_avg_test_metric
            test_results["agg_std_post_last_local_model_avg_across_clients"] = all_std_test_metric
            log(
                INFO,
                f"""Avg Post-aggregation Last Local Model Test
                  Performance Over Aggregated clients: {all_avg_test_metric}""",
            )
            log(
                INFO,
                f"""Std. Dev. Post-aggregation Last Local Model Test
                  Performance Over Aggregated clients: {all_std_test_metric}""",
            )

    if eval_best_global_model:
        all_server_avg_test_metric, all_server_std_test_metric = get_metric_avg_std(
            list(all_best_server_test_metrics.values())
        )
        test_results["avg_best_server_model_avg_across_clients"] = all_server_avg_test_metric
        test_results["std_best_server_model_avg_across_clients"] = all_server_std_test_metric
        log(INFO, f"Avg. Best Server Model Test Performance Over all clients: {all_server_avg_test_metric}")
        log(INFO, f"Std. Dev. Best Server Model Test Performance Over all clients: {all_server_std_test_metric}")

        if eval_over_aggregated_test_data:
            all_server_avg_test_metric, all_server_std_test_metric = get_metric_avg_std(
                list(all_best_server_agg_test_metrics.values())
            )
            test_results["agg_avg_best_server_model_avg_across_clients"] = all_server_avg_test_metric
            test_results["agg_std_best_server_model_avg_across_clients"] = all_server_std_test_metric
            log(
                INFO,
                f"""Avg. Best Server Model Test Performance Over Aggregated
                 clients: {all_server_avg_test_metric}""",
            )
            log(
                INFO,
                f"""Std. Dev. Best Server Model Test Performance Over Aggregated
                  clients: {all_server_std_test_metric}""",
            )

    if eval_last_global_model:
        all_server_avg_test_metric, all_server_std_test_metric = get_metric_avg_std(
            list(all_last_server_test_metrics.values())
        )
        test_results["avg_last_server_model_avg_across_clients"] = all_server_avg_test_metric
        test_results["std_last_server_model_avg_across_clients"] = all_server_std_test_metric
        log(INFO, f"Avg. Last Server Model Test Performance Over all clients: {all_server_avg_test_metric}")
        log(INFO, f"Std. Dev. Last Server Model Test Performance Over all clients: {all_server_std_test_metric}")

        if eval_over_aggregated_test_data:
            all_server_avg_test_metric, all_server_std_test_metric = get_metric_avg_std(
                list(all_last_server_agg_test_metrics.values())
            )
            test_results["agg_avg_last_server_model_avg_across_clients"] = all_server_avg_test_metric
            test_results["agg_std_last_server_model_avg_across_clients"] = all_server_std_test_metric
            log(
                INFO,
                f"""Avg. Last Server Model Test Performance Over Aggregated
                  clients: {all_server_avg_test_metric}""",
            )
            log(
                INFO,
                f"""Std. Dev. Last Server Model Test Performance Over Aggregated
                  clients: {all_server_std_test_metric}""",
            )

    write_measurement_results(eval_write_path, test_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Trained Models on Test Data")
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
        help="Path to the preprocessed Rxrx1 Dataset (ex. path/to/rxrx1)",
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
        "--eval_best_global_model",
        action="store_true",
        help="boolean to indicate whether to search for and evaluate best server model in addition to client models",
    )
    parser.add_argument(
        "--eval_last_global_model",
        action="store_true",
        help="boolean to indicate whether to search for and evaluate last server model in addition to client models",
    )
    parser.add_argument(
        "--eval_best_pre_aggregation_local_models",
        action="store_true",
        help="""boolean to indicate whether to search for and evaluate best pre-aggregation local models in addition
          to the server model""",
    )
    parser.add_argument(
        "--eval_best_post_aggregation_local_models",
        action="store_true",
        help="""boolean to indicate whether to search for and evaluate best post-aggregation local models in addition
          to the server model""",
    )
    parser.add_argument(
        "--eval_last_pre_aggregation_local_models",
        action="store_true",
        help="""boolean to indicate whether to search for and evaluate last pre-aggregation local models in addition
          to the server model""",
    )
    parser.add_argument(
        "--eval_last_post_aggregation_local_models",
        action="store_true",
        help="""boolean to indicate whether to search for and evaluate last post-aggregation local models in addition
          to the server model""",
    )
    parser.add_argument(
        "--eval_over_aggregated_test_data",
        action="store_true",
        help="""boolean to indicate whether to evaluate all the models on the over-aggregated test data as well as
          client specific data""",
    )

    args = parser.parse_args()
    log(INFO, f"Artifact Directory: {args.artifact_dir}")
    log(INFO, f"Dataset Directory: {args.dataset_dir}")
    log(INFO, f"Eval Write Path: {args.eval_write_path}")

    log(INFO, f"Run Best Global Model: {args.eval_best_global_model}")
    log(INFO, f"Run Last Global Model: {args.eval_last_global_model}")
    log(INFO, f"Run Best Pre-aggregation Local Model: {args.eval_best_pre_aggregation_local_models}")
    log(INFO, f"Run Last Pre-aggregation Local Model: {args.eval_last_pre_aggregation_local_models}")
    log(INFO, f"Run Best Post-aggregation Local Model: {args.eval_best_post_aggregation_local_models}")
    log(INFO, f"Run Last Post-aggregation Local Model: {args.eval_last_post_aggregation_local_models}")
    log(INFO, f"Run Eval Over Aggregated Test Data: {args.eval_over_aggregated_test_data}")

    assert (
        args.eval_best_global_model
        or args.eval_last_global_model
        or args.eval_best_pre_aggregation_local_models
        or args.eval_last_pre_aggregation_local_models
        or args.eval_best_post_aggregation_local_models
        or args.eval_last_post_aggregation_local_models
    )
    main(
        args.artifact_dir,
        args.dataset_dir,
        args.eval_write_path,
        args.eval_best_pre_aggregation_local_models,
        args.eval_last_pre_aggregation_local_models,
        args.eval_best_post_aggregation_local_models,
        args.eval_last_post_aggregation_local_models,
        args.eval_best_global_model,
        args.eval_last_global_model,
        args.eval_over_aggregated_test_data,
    )
