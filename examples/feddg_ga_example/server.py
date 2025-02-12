import argparse
from functools import partial
from typing import Any

import flwr as fl
from flwr.common.typing import Config

from examples.models.cnn_model import MnistNetWithBnAndFrozen
from fl4health.client_managers.fixed_sampling_client_manager import FixedSamplingClientManager
from fl4health.model_bases.apfl_base import ApflModule
from fl4health.reporting import JsonReporter
from fl4health.servers.base_server import FlServer
from fl4health.strategies.feddg_ga import FedDgGa
from fl4health.utils.config import load_config, make_dict_with_epochs_or_steps
from fl4health.utils.metric_aggregation import evaluate_metrics_aggregation_fn, fit_metrics_aggregation_fn
from fl4health.utils.parameter_extraction import get_all_model_parameters
from fl4health.utils.random import set_all_random_seeds


def fit_config(
    batch_size: int,
    n_server_rounds: int,
    current_round: int,
    local_epochs: int | None = None,
    local_steps: int | None = None,
    evaluate_after_fit: bool = False,
    pack_losses_with_val_metrics: bool = False,
) -> Config:
    return {
        **make_dict_with_epochs_or_steps(local_epochs, local_steps),
        "current_server_round": current_round,
        "batch_size": batch_size,
        "n_server_rounds": n_server_rounds,
        "evaluate_after_fit": evaluate_after_fit,
        "pack_losses_with_val_metrics": pack_losses_with_val_metrics,
    }


def main(config: dict[str, Any]) -> None:
    # This function will be used to produce a config that is sent to each client to initialize their own environment
    fit_config_fn = partial(
        fit_config,
        config["batch_size"],
        config["n_server_rounds"],
        local_epochs=config.get("local_epochs"),
        local_steps=config.get("local_steps"),
        evaluate_after_fit=config.get("evaluate_after_fit", False),
        pack_losses_with_val_metrics=config.get("pack_losses_with_val_metrics", False),
    )

    initial_model = ApflModule(MnistNetWithBnAndFrozen())

    # Implementation of FedDG-GA as a server side strategy
    strategy = FedDgGa(
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
    )

    # FixedSamplingClientManager is a requirement here because the sampling cannot
    # be different between validation and evaluation for FedDG-GA to work. FixedSamplingClientManager
    # will return the same sampling until it is told to reset, which in FedDgGaStrategy
    # is done right before fit_round.
    client_manager = FixedSamplingClientManager()
    server = FlServer(
        client_manager=client_manager,
        fl_config=config,
        strategy=strategy,
        reporters=[JsonReporter()],
        accept_failures=False,
    )

    fl.server.start_server(
        server=server,
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=config["n_server_rounds"]),
    )

    server.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FL Server Main")
    parser.add_argument(
        "--config_path",
        action="store",
        type=str,
        help="Path to configuration file.",
        default="examples/apfl_example/config.yaml",
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

    # Set the random seed for reproducibility
    set_all_random_seeds(args.seed)

    main(config)
