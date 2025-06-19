import argparse
from functools import partial
from logging import INFO
from typing import Any

import flwr as fl
from flwr.common.logger import log
from flwr.common.typing import Config
from flwr.server.client_manager import SimpleClientManager
from torch import nn

from examples.models.cnn_model import MnistNetWithBnAndFrozen, SkinCancerNet
from fl4health.metrics.metric_aggregation import evaluate_metrics_aggregation_fn, fit_metrics_aggregation_fn
from fl4health.servers.base_server import FlServer
from fl4health.strategies.basic_fedavg import BasicFedAvg
from fl4health.utils.config import load_config, make_dict_with_epochs_or_steps
from fl4health.utils.parameter_extraction import get_all_model_parameters


def fit_config(
    batch_size: int,
    n_server_rounds: int,
    current_round: int,
    local_epochs: int | None = None,
    local_steps: int | None = None,
) -> Config:
    return {
        **make_dict_with_epochs_or_steps(local_epochs, local_steps),
        "batch_size": batch_size,
        "n_server_rounds": n_server_rounds,
        "current_server_round": current_round,
    }


def main(config: dict[str, Any], server_address: str, dataset_name: str) -> None:
    # This function will be used to produce a config that is sent to each client to initialize their own environment
    fit_config_fn = partial(
        fit_config,
        config["batch_size"],
        config["n_server_rounds"],
        local_epochs=config.get("local_epochs"),
        local_steps=config.get("local_steps"),
    )

    if dataset_name in ["Barcelona", "Rosendahl", "Vienna", "UFES", "Canada"]:
        initial_model: nn.Module = SkinCancerNet()
    elif dataset_name == "mnist":
        initial_model = MnistNetWithBnAndFrozen(freeze_cnn_layer=False)
    else:
        raise ValueError(
            "Unsupported location. Please choose from 'Barcelona', 'Rosendahl', \
            'Vienna', 'UFES', 'Canada', or 'mnist'."
        )

    # Server performs simple FedAveraging as its server-side optimization strategy
    strategy = BasicFedAvg(
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
        weighted_aggregation=True,
        weighted_eval_losses=True,
    )

    client_manager = SimpleClientManager()
    server = FlServer(client_manager=client_manager, fl_config=config, strategy=strategy, accept_failures=False)

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
        "--dataset_name",
        action="store",
        type=str,
        help="Dataset name (e.g., Barcelona, Rosendahl, Vienna, UFES, Canada for Skin Cancer; \
            'mnist' for MNIST dataset)",
        default="mnist",
    )
    args = parser.parse_args()

    config = load_config(args.config_path)
    log(INFO, f"Server Address: {args.server_address}")
    main(config, args.server_address, args.dataset_name)
