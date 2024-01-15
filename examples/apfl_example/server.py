import argparse
from functools import partial
from typing import Any, Dict, Optional

import flwr as fl
from flwr.common.parameter import ndarrays_to_parameters
from flwr.common.typing import Config, Parameters
from flwr.server.client_manager import SimpleClientManager
from flwr.server.strategy import FedAvg

from examples.models.cnn_model import MnistNetWithBnAndFrozen
from examples.simple_metric_aggregation import evaluate_metrics_aggregation_fn, fit_metrics_aggregation_fn
from examples.utils.functions import make_dict_with_epochs_or_steps
from fl4health.model_bases.apfl_base import ApflModule
from fl4health.server.base_server import FlServer
from fl4health.utils.config import load_config
from fl4health.utils.random import set_all_random_seeds


def get_initial_model_parameters() -> Parameters:
    # Initializing the model parameters on the server side.
    # Currently uses the Pytorch default initialization for the model parameters.
    initial_model = ApflModule(MnistNetWithBnAndFrozen())
    return ndarrays_to_parameters([val.cpu().numpy() for _, val in initial_model.state_dict().items()])


def fit_config(
    batch_size: int,
    n_server_rounds: int,
    current_round: int,
    local_epochs: Optional[int] = None,
    local_steps: Optional[int] = None,
) -> Config:
    return {
        **make_dict_with_epochs_or_steps(local_epochs, local_steps),
        "current_server_round": current_round,
        "batch_size": batch_size,
        "n_server_rounds": n_server_rounds,
    }


def main(config: Dict[str, Any]) -> None:
    # This function will be used to produce a config that is sent to each client to initialize their own environment
    fit_config_fn = partial(
        fit_config,
        config["batch_size"],
        config["n_server_rounds"],
        local_epochs=config.get("local_epochs"),
        local_steps=config.get("local_steps"),
    )

    # Server performs simple FedAveraging as its server-side optimization strategy
    strategy = FedAvg(
        min_fit_clients=config["n_clients"],
        min_evaluate_clients=config["n_clients"],
        # Server waits for min_available_clients before starting FL rounds
        min_available_clients=config["n_clients"],
        on_fit_config_fn=fit_config_fn,
        # We use the same fit config function, as nothing changes for eval
        on_evaluate_config_fn=fit_config_fn,
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        initial_parameters=get_initial_model_parameters(),
    )

    client_manager = SimpleClientManager()
    server = FlServer(client_manager, strategy)

    fl.server.start_server(
        server=server,
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=config["n_server_rounds"]),
    )

    server.metrics_reporter.dump()


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
