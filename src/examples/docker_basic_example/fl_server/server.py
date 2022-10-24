import os
from typing import List, Tuple

import flwr as fl
from flwr.common.typing import Metrics
from flwr.server.strategy import FedAvg

NUM_CLIENTS = int(os.getenv("NUM_CLIENTS"))  # type: ignore
NUM_ROUNDS = int(os.getenv("NUM_ROUNDS"))  # type: ignore
SERVER_INTERNAL_HOST = os.getenv("SERVER_INTERNAL_HOST")
SERVER_INTERNAL_PORT = os.getenv("SERVER_INTERNAL_PORT")


def metric_aggregation(all_client_metrics: List[Tuple[int, Metrics]]) -> Tuple[int, Metrics]:
    aggregated_metrics: Metrics = {}
    total_examples = 0
    # Run through all of the metrics
    for num_examples_on_client, client_metrics in all_client_metrics:
        total_examples += num_examples_on_client
        for metric_name, metric_value in client_metrics.items():
            # Here we assume each metric is normalized by the number of examples on the client. So we scale up to
            # get the "raw" value
            if metric_name in aggregated_metrics:
                aggregated_metrics[metric_name] += num_examples_on_client * metric_value
            else:
                aggregated_metrics[metric_name] = num_examples_on_client * metric_value
    return total_examples, aggregated_metrics


def normalize_metrics(total_examples: int, aggregated_metrics: Metrics) -> Metrics:
    # Normalize all metric values by the total count of examples seen.
    return {metric_name: metric_value / total_examples for metric_name, metric_value in aggregated_metrics.items()}


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


def main() -> None:
    # Server performs simple FedAveraging as it's server-side optimization strategy
    strategy = FedAvg(
        min_fit_clients=NUM_CLIENTS,
        min_evaluate_clients=NUM_CLIENTS,
        # Server waits for min_available_clients before starting FL rounds
        min_available_clients=NUM_CLIENTS,
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
    )
    fl.server.start_server(
        server_address=f"{SERVER_INTERNAL_HOST}:{SERVER_INTERNAL_PORT}",
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
