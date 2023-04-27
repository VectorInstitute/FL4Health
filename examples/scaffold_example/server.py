import argparse
from functools import partial
from typing import Any, Dict, List, Tuple

import flwr as fl
import numpy as np
from flwr.common.parameter import ndarrays_to_parameters
from flwr.common.typing import Config, Metrics, Parameters

from examples.models.cnn_model import MnistNet
from examples.simple_metric_aggregation import metric_aggregation, normalize_metrics
from fl4health.client_managers.poisson_sampling_manager import PoissonSamplingClientManager
from fl4health.strategies.scaffold import Scaffold
from fl4health.utils.config import load_config


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


def get_initial_model_parameters() -> Parameters:
    # Initializing the model parameters on the server side.
    # Currently uses the Pytorch default initialization for the model parameters.
    initial_model = MnistNet()
    model_weights = [val.cpu().numpy() for _, val in initial_model.state_dict().items()]
    control_variates = [np.zeros_like(weight) for weight in model_weights]
    parameters = ndarrays_to_parameters(model_weights + control_variates)
    return parameters


def fit_config(local_steps: int, batch_size: int, n_server_rounds: int, current_round: int) -> Config:
    return {
        "local_steps": local_steps,
        "batch_size": batch_size,
        "n_server_rounds": n_server_rounds,
    }


def main(config: Dict[str, Any]) -> None:
    # This function will be used to produce a config that is sent to each client to initialize their own environment
    fit_config_fn = partial(
        fit_config,
        config["local_steps"],
        config["batch_size"],
        config["n_server_rounds"],
    )

    # Server performs simple FedAveraging as its server-side optimization strategy
    strategy = Scaffold(
        min_available_clients=config["n_clients"],
        on_fit_config_fn=fit_config_fn,
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        initial_parameters=get_initial_model_parameters(),
    )

    # ClientManager that performs Poisson type sampling
    client_manager = PoissonSamplingClientManager()

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
