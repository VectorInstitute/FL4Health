import argparse
from functools import partial
from logging import INFO
from typing import Any, Dict, List, Tuple

import flwr as fl
from flwr.common.logger import log
from flwr.common.parameter import ndarrays_to_parameters
from flwr.common.typing import Config, Metrics, Parameters
from flwr.server.client_manager import SimpleClientManager

from examples.models.cnn_model import MnistNet
from examples.simple_metric_aggregation import metric_aggregation, normalize_metrics
from fl4health.reporting.fl_wanb import ServerWandBReporter
from fl4health.server.base_server import FlServer
from fl4health.strategies.fedprox import FedProx
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


def get_initial_model_information() -> Parameters:
    # Initializing the model parameters on the server side.
    # Currently uses the Pytorch default initialization for the model parameters.
    initial_model = MnistNet()
    model_weights = [val.cpu().numpy() for _, val in initial_model.state_dict().items()]
    return ndarrays_to_parameters(model_weights)


def fit_config(
    local_epochs: int,
    batch_size: int,
    n_server_rounds: int,
    adaptive_proximal_weight: bool,
    proximal_weight: float,
    proximal_weight_delta: float,
    proximal_weight_patience: int,
    reporting_enabled: bool,
    project_name: str,
    group_name: str,
    entity: str,
    current_round: int,
) -> Config:
    return {
        "local_epochs": local_epochs,
        "batch_size": batch_size,
        "n_server_rounds": n_server_rounds,
        "adaptive_proximal_weight": adaptive_proximal_weight,
        "proximal_weight": proximal_weight,
        "proximal_weight_delta": proximal_weight_delta,
        "proximal_weight_patience": proximal_weight_patience,
        "current_server_round": current_round,
        "reporting_enabled": reporting_enabled,
        "project_name": project_name,
        "group_name": group_name,
        "entity": entity,
    }


def main(config: Dict[str, Any], server_address: str) -> None:
    # This function will be used to produce a config that is sent to each client to initialize their own environment
    fit_config_fn = partial(
        fit_config,
        config["local_epochs"],
        config["batch_size"],
        config["n_server_rounds"],
        config["adaptive_proximal_weight"],
        config["proximal_weight"],
        config["proximal_weight_delta"],
        config["proximal_weight_patience"],
        config["reporting_config"].get("enabled", False),
        # Note that run name is not included, it will be set in the clients
        config["reporting_config"].get("project_name", ""),
        config["reporting_config"].get("group_name", ""),
        config["reporting_config"].get("entity", ""),
    )

    initial_parameters = get_initial_model_information()

    # Server performs simple FedAveraging as its server-side optimization strategy
    strategy = FedProx(
        min_fit_clients=config["n_clients"],
        min_evaluate_clients=config["n_clients"],
        # Server waits for min_available_clients before starting FL rounds
        min_available_clients=config["n_clients"],
        on_fit_config_fn=fit_config_fn,
        # We use the same fit config function, as nothing changes for eval
        on_evaluate_config_fn=fit_config_fn,
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        initial_parameters=initial_parameters,
        adaptive_proximal_weight=config["adaptive_proximal_weight"],
        proximal_weight=config["proximal_weight"],
        proximal_weight_delta=config["proximal_weight_delta"],
        proximal_weight_patience=config["proximal_weight_patience"],
    )

    wandb_reporter = ServerWandBReporter.from_config(config)
    client_manager = SimpleClientManager()
    server = FlServer(client_manager, strategy, wandb_reporter)

    fl.server.start_server(
        server=server,
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=config["n_server_rounds"]),
    )
    # Shutdown the server gracefully
    server.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FL Server Main")
    parser.add_argument(
        "--config_path",
        action="store",
        type=str,
        help="Path to configuration file.",
        default="examples/fedprox_example/config.yaml",
    )
    parser.add_argument(
        "--server_address",
        action="store",
        type=str,
        help="Server Address to be used to communicate with the clients",
        default="0.0.0.0:8080",
    )
    args = parser.parse_args()

    config = load_config(args.config_path)
    log(INFO, f"Server Address: {args.server_address}")
    main(config, args.server_address)
