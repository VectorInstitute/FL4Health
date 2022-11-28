from functools import partial
from logging import INFO
from typing import List, Tuple

import flwr as fl
from flwr.common.logger import log
from flwr.common.parameter import ndarrays_to_parameters
from flwr.common.typing import Config, Metrics, Parameters

from src.client_managers.poisson_sampling_manager import PoissonSamplingClientManager
from src.examples.dp_fed_examples.client_level_dp.model import Net
from src.privacy.fl_accountants import FlClientLevelAccountantPoissonSampling
from src.strategies.client_dp_fedavgm import ClientLevelDPFedAvgM


def get_initial_model_parameters() -> Parameters:
    # The server-side strategy requires that we provide server side parameter initialization.
    # Currently uses the Pytorch default initialization for the model parameters.
    initial_model = Net()
    return ndarrays_to_parameters([val.cpu().numpy() for _, val in initial_model.state_dict().items()])


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
) -> Config:
    # NOTE: The omitted variable is server_round which allows for dynamically changing the config each round
    return {
        "local_epochs": local_epochs,
        "batch_size": batch_size,
    }


def fit_config(
    local_epochs: int,
    batch_size: int,
    server_round: int,
) -> Config:
    return construct_config(server_round, local_epochs, batch_size)


def main() -> None:

    # Server parameters
    NUM_SERVER_ROUNDS = 20
    SERVER_NOISE_MULTIPLIER = 0.01
    NUM_CLIENTS = 3
    CLIENT_SAMPLING = 2.0 / NUM_CLIENTS
    SERVER_LEARNING_RATE = 1.0
    SERVER_MOMENTUM = 0.2

    # Client training parameters
    CLIENT_EPOCHS = 1
    CLIENT_BATCH_SIZE = 128

    # Clipping settings for update and optionally
    # adaptive clipping
    ADAPTIVE_CLIPPING = True
    CLIPPING_BOUND = 0.1
    CLIPPING_LEARNING_RATE = 0.5
    CLIPPING_BIT_NOISE_MULTIPLIER = 0.5
    CLIPPING_QUANTILE = 0.5

    # This function will be used to produce a config that is sent to each client to initialize their own environment
    fit_config_fn = partial(fit_config, CLIENT_EPOCHS, CLIENT_BATCH_SIZE)

    # ClientManager that performs Poisson type sampling
    client_manager = PoissonSamplingClientManager()

    # Accountant that computes the privacy through training
    accountant = FlClientLevelAccountantPoissonSampling(CLIENT_SAMPLING, SERVER_NOISE_MULTIPLIER)
    target_delta = 1.0 / NUM_CLIENTS
    epsilon = accountant.get_epsilon(NUM_SERVER_ROUNDS, target_delta)
    log(INFO, f"Model privacy after full training will be ({epsilon}, {target_delta})")

    # Server performs simple FedAveraging as it's server-side optimization strategy
    strategy = ClientLevelDPFedAvgM(
        fraction_fit=CLIENT_SAMPLING,
        # Server waits for min_available_clients before starting FL rounds
        min_available_clients=NUM_CLIENTS,
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        on_fit_config_fn=fit_config_fn,
        on_evaluate_config_fn=fit_config_fn,
        # Server side weight initialization
        initial_parameters=get_initial_model_parameters(),
        adaptive_clipping=ADAPTIVE_CLIPPING,
        server_learning_rate=SERVER_LEARNING_RATE,
        clipping_learning_rate=CLIPPING_LEARNING_RATE,
        clipping_quantile=CLIPPING_QUANTILE,
        initial_clipping_bound=CLIPPING_BOUND,
        weight_noise_multiplier=SERVER_NOISE_MULTIPLIER,
        clipping_noise_mutliplier=CLIPPING_BIT_NOISE_MULTIPLIER,
        beta=SERVER_MOMENTUM,
    )

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=NUM_SERVER_ROUNDS),
        strategy=strategy,
        client_manager=client_manager,
    )


if __name__ == "__main__":
    main()
