import argparse
from functools import partial
from typing import Any

import flwr as fl
from flwr.common.typing import Config

from examples.models.cnn_model import MnistNetWithBnAndFrozen
from fl4health.client_managers.poisson_sampling_manager import PoissonSamplingClientManager
from fl4health.metrics.metric_aggregation import evaluate_metrics_aggregation_fn, fit_metrics_aggregation_fn
from fl4health.reporting import JsonReporter
from fl4health.servers.scaffold_server import ScaffoldServer
from fl4health.strategies.scaffold import Scaffold
from fl4health.utils.config import load_config
from fl4health.utils.parameter_extraction import get_all_model_parameters
from fl4health.utils.random import set_all_random_seeds


def fit_config(local_steps: int, batch_size: int, n_server_rounds: int, current_round: int) -> Config:
    return {
        "local_steps": local_steps,
        "batch_size": batch_size,
        "n_server_rounds": n_server_rounds,
        "current_server_round": current_round,
    }


def main(config: dict[str, Any]) -> None:
    # This function will be used to produce a config that is sent to each client to initialize their own environment
    fit_config_fn = partial(
        fit_config,
        config["local_steps"],
        config["batch_size"],
        config["n_server_rounds"],
    )

    model = MnistNetWithBnAndFrozen()

    # Initialize Scaffold strategy to handle aggregation of weights and corresponding control variates
    strategy = Scaffold(
        min_available_clients=config["n_clients"],
        on_fit_config_fn=fit_config_fn,
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        # We use the same fit config function, as nothing changes for eval
        on_evaluate_config_fn=fit_config_fn,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        initial_parameters=get_all_model_parameters(model),
        # Sending the model allows for the creation of the initial control variates
        model=model,
    )

    # ClientManager that performs Poisson type sampling
    client_manager = PoissonSamplingClientManager()

    server = ScaffoldServer(
        client_manager=client_manager,
        fl_config=config,
        strategy=strategy,
        warm_start=True,
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
        default="examples/scaffold_example/config.yaml",
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
