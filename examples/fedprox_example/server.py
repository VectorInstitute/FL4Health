import argparse
from functools import partial
from logging import INFO
from typing import Any, Dict

import flwr as fl
from flwr.common.logger import log
from flwr.common.typing import Config
from flwr.server.client_manager import SimpleClientManager

from examples.models.cnn_model import MnistNet
from examples.utils.functions import make_dict_with_epochs_or_steps
from fl4health.reporting import JsonReporter, WandBReporter
from fl4health.servers.adaptive_constraint_servers.fedprox_server import FedProxServer
from fl4health.strategies.fedavg_with_adaptive_constraint import FedAvgWithAdaptiveConstraint
from fl4health.utils.config import load_config
from fl4health.utils.metric_aggregation import evaluate_metrics_aggregation_fn, fit_metrics_aggregation_fn
from fl4health.utils.parameter_extraction import get_all_model_parameters
from fl4health.utils.random import set_all_random_seeds


def fit_config(
    batch_size: int,
    n_server_rounds: int,
    current_round: int,
    reporting_config: Dict[str, str] | None = None,
    local_epochs: int | None = None,
    local_steps: int | None = None,
) -> Config:
    base_config: Config = {
        **make_dict_with_epochs_or_steps(local_epochs, local_steps),
        "batch_size": batch_size,
        "n_server_rounds": n_server_rounds,
        "current_server_round": current_round,
    }
    if reporting_config is not None:
        # NOTE: that name is not included, it will be set in the clients
        base_config["project"] = reporting_config.get("project", "")
        base_config["group"] = reporting_config.get("group", "")
        base_config["entity"] = reporting_config.get("entity", "")

    return base_config


def main(config: Dict[str, Any], server_address: str) -> None:
    # This function will be used to produce a config that is sent to each client to initialize their own environment
    fit_config_fn = partial(
        fit_config,
        config["batch_size"],
        config["n_server_rounds"],
        reporting_config=config.get("reporting_config"),
        local_epochs=config.get("local_epochs"),
        local_steps=config.get("local_steps"),
    )

    initial_model = MnistNet()

    # Server performs simple FedAveraging as its server-side optimization strategy and potentially adapts the
    # FedProx proximal weight mu
    strategy = FedAvgWithAdaptiveConstraint(
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
        adapt_loss_weight=config["adapt_proximal_weight"],
        initial_loss_weight=config["initial_proximal_weight"],
        loss_weight_delta=config["proximal_weight_delta"],
        loss_weight_patience=config["proximal_weight_patience"],
    )

    json_reporter = JsonReporter()
    client_manager = SimpleClientManager()
    if "reporting_config" in config:
        wandb_reporter = WandBReporter("round", **config["reporting_config"])
        reporters = [wandb_reporter, json_reporter]
    else:
        reporters = [json_reporter]
    server = FedProxServer(client_manager=client_manager, fl_config=config, strategy=strategy, reporters=reporters)

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
