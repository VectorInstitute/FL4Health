from functools import partial
from logging import INFO
from typing import List, Tuple

import flwr as fl
from flwr.common.logger import log
from flwr.common.typing import Config, Metrics

from src.client_managers.poisson_sampling_manager import PoissonSamplingClientManager
from src.privacy.fl_accountants import FlInstanceLevelAccountant
from src.strategies.fedavg import FedAvgSampling


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


def fit_metrics_aggregation_fn(
    all_client_metrics: List[Tuple[int, Metrics]],
) -> Metrics:
    # This function is run by the server to aggregate metrics returned by each clients fit function
    # NOTE: The first value of the tuple is number of examples for FedAvg
    total_examples, aggregated_metrics = metric_aggregation(all_client_metrics)
    return normalize_metrics(total_examples, aggregated_metrics)


def evaluate_metrics_aggregation_fn(all_client_metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # This function is run by the server to aggregate metrics returned by each clients evaluate function
    # NOTE: The first value of the tuple is number of examples for FedAvg
    total_examples, aggregated_metrics = metric_aggregation(all_client_metrics)
    return normalize_metrics(total_examples, aggregated_metrics)


def construct_config(
    _: int,
    local_epochs: int,
    batch_size: int,
    noise_multiplier: float,
    clipping_bound: float,
) -> Config:
    # NOTE: a new client is created in each round
    # NOTE: The omitted variable is server_round which allows for dynamically changing the config each round
    return {
        "local_epochs": local_epochs,
        "batch_size": batch_size,
        "noise_multiplier": noise_multiplier,
        "clipping_bound": clipping_bound,
    }


def fit_config(
    local_epochs: int,
    batch_size: int,
    noise_multiplier: float,
    clipping_bound: float,
    server_round: int,
) -> Config:
    return construct_config(server_round, local_epochs, batch_size, noise_multiplier, clipping_bound)


def main() -> None:

    NUM_SERVER_ROUNDS = 5

    NUM_CLIENTS = 3
    CLIENT_SAMPLING = 2.0 / 3.0
    CLIENT_EPOCHS = 4

    CLIENT_NOISE_MULTIPLIER = 1.0
    ClIENT_CLIPPING = 5.0

    CLIENT_BATCH_SIZE = 64
    CLIENT_DATA_SIZES = 50000
    TOTAL_DATA_SIZE = CLIENT_DATA_SIZES * 3

    # This function will be used to produce a config that is sent to each client to initialize their own environment
    fit_config_fn = partial(fit_config, CLIENT_EPOCHS, CLIENT_BATCH_SIZE, CLIENT_NOISE_MULTIPLIER, ClIENT_CLIPPING)

    # ClientManager that performs Poisson type sampling
    client_manager = PoissonSamplingClientManager()

    # Accountant that computes the privacy through training
    accountant = FlInstanceLevelAccountant(
        CLIENT_SAMPLING, CLIENT_NOISE_MULTIPLIER, CLIENT_EPOCHS, [CLIENT_BATCH_SIZE], [CLIENT_DATA_SIZES]
    )

    # Server performs simple FedAveraging as it's server-side optimization strategy
    strategy = FedAvgSampling(
        fraction_fit=CLIENT_SAMPLING,
        # Server waits for min_available_clients before starting FL rounds
        min_available_clients=NUM_CLIENTS,
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        on_fit_config_fn=fit_config_fn,
        on_evaluate_config_fn=fit_config_fn,
    )

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=NUM_SERVER_ROUNDS),
        strategy=strategy,
        client_manager=client_manager,
    )

    target_delta = 1.0 / TOTAL_DATA_SIZE
    epsilon = accountant.get_epsilon(NUM_SERVER_ROUNDS, target_delta)
    log(INFO, f"Privacy ({epsilon}, {target_delta})")


if __name__ == "__main__":
    main()
