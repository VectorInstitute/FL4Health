import argparse
from functools import partial
from logging import INFO
from typing import Any, Dict, List, Tuple

import flwr as fl
from flwr.common.logger import log
from flwr.common.parameter import ndarrays_to_parameters
from flwr.common.typing import Config, Metrics, Parameters

from examples.models.cnn_model import Net
from examples.simple_metric_aggregation import metric_aggregation, normalize_metrics
from fl4health.client_managers.poisson_sampling_manager import PoissonSamplingClientManager
from fl4health.privacy.fl_accountants import FlInstanceLevelAccountant
from fl4health.strategies.fedavg_sampling import FedAvgSampling
from fl4health.utils.config import load_config


def get_initial_model_parameters() -> Parameters:
    # The server-side strategy requires that we provide server side parameter initialization.
    # Currently uses the Pytorch default initialization for the model parameters.
    initial_model = Net()
    return ndarrays_to_parameters([val.cpu().numpy() for _, val in initial_model.state_dict().items()])


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
    current_round: int,
) -> Config:
    return construct_config(current_round, local_epochs, batch_size, noise_multiplier, clipping_bound)


def main(config: Dict[str, Any]) -> None:
    # This function will be used to produce a config that is sent to each client to initialize their own environment
    fit_config_fn = partial(
        fit_config,
        config["local_epochs"],
        config["batch_size"],
        config["client_noise_multiplier"],
        config["client_clipping"],
    )

    # ClientManager that performs Poisson type sampling
    client_manager = PoissonSamplingClientManager()

    # Accountant that computes the privacy through training
    accountant = FlInstanceLevelAccountant(
        config["client_sampling_rate"],
        config["client_noise_multiplier"],
        config["local_epochs"],
        [config["batch_size"]],
        [config["client_data_sizes"]],
    )
    target_delta = 1.0 / config["total_data_size"]
    epsilon = accountant.get_epsilon(config["n_server_rounds"], target_delta)
    log(INFO, f"Model privacy after full training will be ({epsilon}, {target_delta})")

    # Server performs simple FedAveraging as it's server-side optimization strategy
    strategy = FedAvgSampling(
        fraction_fit=config["client_sampling_rate"],
        # Server waits for min_available_clients before starting FL rounds
        min_available_clients=config["n_clients"],
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        on_fit_config_fn=fit_config_fn,
        on_evaluate_config_fn=fit_config_fn,
        # Server side weight initialization
        initial_parameters=get_initial_model_parameters(),
    )

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=config["n_server_rounds"]),
        strategy=strategy,
        client_manager=client_manager,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FL Server Main")
    parser.add_argument(
        "--config_path",
        action="store",
        type=str,
        help="Path to configuration file.",
        default="config.yaml",
    )
    args = parser.parse_args()

    config = load_config(args.config_path)
    main(config)
