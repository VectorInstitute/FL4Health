import argparse
from functools import partial
from logging import INFO
from typing import Any, Dict, Optional

import flwr as fl
from flwr.common.logger import log
from flwr.common.typing import Config
from flwr.server.client_manager import SimpleClientManager

from examples.models.cnn_model import MnistNet
from examples.utils.functions import make_dict_with_epochs_or_steps
from fl4health.reporting.fl_wandb import ServerWandBReporter
from fl4health.server.base_server import FlServer
from fl4health.strategies.fedprox import FedProx
from fl4health.utils.config import load_config
from fl4health.utils.functions import get_all_model_parameters
from fl4health.utils.metric_aggregation import evaluate_metrics_aggregation_fn, fit_metrics_aggregation_fn
from fl4health.utils.random import set_all_random_seeds


def fit_config(
    batch_size: int,
    n_server_rounds: int,
    reporting_enabled: bool,
    project_name: str,
    group_name: str,
    entity: str,
    current_round: int,
    local_epochs: Optional[int] = None,
    local_steps: Optional[int] = None,
) -> Config:
    return {
        **make_dict_with_epochs_or_steps(local_epochs, local_steps),
        "batch_size": batch_size,
        "n_server_rounds": n_server_rounds,
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
        config["batch_size"],
        config["n_server_rounds"],
        config["reporting_config"].get("enabled", False),
        # Note that run name is not included, it will be set in the clients
        config["reporting_config"].get("project_name", ""),
        config["reporting_config"].get("group_name", ""),
        config["reporting_config"].get("entity", ""),
        local_epochs=config.get("local_epochs"),
        local_steps=config.get("local_steps"),
    )

    initial_model = MnistNet()

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
        initial_parameters=get_all_model_parameters(initial_model),
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
        default="examples/warm_up_examples/warmed_up_fedprox/config.yaml",
    )
    parser.add_argument(
        "--server_address",
        action="store",
        type=str,
        help="Server Address to be used to communicate with the clients",
        default="0.0.0.0:8080",
    )
    parser.add_argument(
        "--seed",
        action="store",
        type=int,
        help="Seed for the random number generators across python, torch, and numpy",
        required=False,
    )
    args = parser.parse_args()

    config = load_config(args.config_path)
    log(INFO, f"Server Address: {args.server_address}")

    # Set the random seed for reproducibility
    set_all_random_seeds(args.seed)

    main(config, args.server_address)
