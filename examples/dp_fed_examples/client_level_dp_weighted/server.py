import argparse
import pickle
from functools import partial
from logging import INFO
from typing import Any, Dict, List, Tuple

import flwr as fl
from flwr.common.logger import log
from flwr.common.parameter import ndarrays_to_parameters
from flwr.common.typing import Config, Metrics, Parameters

from examples.dp_fed_examples.client_level_dp_weighted.data import Scaler
from examples.models.logistic_regression import LogisticRegression
from fl4health.client_managers.poisson_sampling_manager import PoissonSamplingClientManager
from fl4health.privacy.fl_accountants import FlClientLevelAccountantPoissonSampling
from fl4health.strategies.client_dp_fedavgm import ClientLevelDPFedAvgM
from fl4health.utils.config import load_config


def get_initial_model_parameters() -> Parameters:
    # The server-side strategy requires that we provide server side parameter initialization.
    # Currently uses the Pytorch default initialization for the model parameters.
    initial_model = LogisticRegression()
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


def construct_config(_: int, local_epochs: int, batch_size: int, adaptive_clipping: bool) -> Config:
    # NOTE: The omitted variable is server_round which allows for dynamically changing the config each round
    return {
        "local_epochs": local_epochs,
        "batch_size": batch_size,
        "adaptive_clipping": adaptive_clipping,
        "scaler": pickle.dumps(Scaler()),
    }


def fit_config(
    local_epochs: int,
    batch_size: int,
    adaptive_clipping: bool,
    server_round: int,
) -> Config:
    return construct_config(server_round, local_epochs, batch_size, adaptive_clipping)


def main(config: Dict[str, Any]) -> None:

    # This function will be used to produce a config that is sent to each client to initialize their own environment
    fit_config_fn = partial(fit_config, config["local_epochs"], config["batch_size"], config["adaptive_clipping"])

    # ClientManager that performs Poisson type sampling
    client_manager = PoissonSamplingClientManager()

    # Accountant that computes the privacy through training
    accountant = FlClientLevelAccountantPoissonSampling(config["client_sampling"], config["server_noise_multiplier"])
    target_delta = 1.0 / config["n_clients"]
    epsilon = accountant.get_epsilon(config["n_server_rounds"], target_delta)
    log(INFO, f"Model privacy after full training will be ({epsilon}, {target_delta})")

    # Server performs simple FedAveraging as it's server-side optimization strategy
    strategy = ClientLevelDPFedAvgM(
        fraction_fit=config["client_sampling"],
        # Server waits for min_available_clients before starting FL rounds
        min_available_clients=config["n_clients"],
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        on_fit_config_fn=fit_config_fn,
        on_evaluate_config_fn=fit_config_fn,
        # Server side weight initialization
        initial_parameters=get_initial_model_parameters(),
        adaptive_clipping=config["adaptive_clipping"],
        server_learning_rate=config["server_learning_rate"],
        clipping_learning_rate=config["clipping_learning_rate"],
        clipping_quantile=config["clipping_quantile"],
        initial_clipping_bound=config["clipping_bound"],
        weight_noise_multiplier=config["server_noise_multiplier"],
        clipping_noise_mutliplier=config["clipping_bit_noise_multiplier"],
        beta=config["server_momentum"],
        weighted_averaging=config["weighted_averaging"],
        total_samples=config["total_samples"],
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
