import argparse
import os
from functools import partial
from logging import INFO
from typing import Any, Dict

import flwr as fl
from flamby.datasets.fed_isic2019 import Baseline
from flwr.common.logger import log
from flwr.server.client_manager import SimpleClientManager

from fl4health.checkpointing.checkpointer import BestMetricTorchCheckpointer
from fl4health.strategies.scaffold import Scaffold
from fl4health.utils.config import load_config
from research.flamby.flamby_servers.scaffold_server import ScaffoldServer
from research.flamby.utils import (
    evaluate_metrics_aggregation_fn,
    fit_config,
    fit_metrics_aggregation_fn,
    get_initial_model_info_with_control_variates,
)


def main(
    config: Dict[str, Any], server_address: str, checkpoint_stub: str, run_name: str, server_learning_rate: float
) -> None:
    # This function will be used to produce a config that is sent to each client to initialize their own environment
    fit_config_fn = partial(
        fit_config,
        config["local_steps"],
        config["n_server_rounds"],
    )

    checkpoint_dir = os.path.join(checkpoint_stub, run_name)
    checkpoint_name = "server_best_model.pkl"
    checkpointer = BestMetricTorchCheckpointer(checkpoint_dir, checkpoint_name)

    client_manager = SimpleClientManager()
    client_model = Baseline()

    initial_parameters, initial_control_variates = get_initial_model_info_with_control_variates(client_model)

    strategy = Scaffold(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        # Server waits for min_available_clients before starting FL rounds
        min_available_clients=config["n_clients"],
        on_fit_config_fn=fit_config_fn,
        # We use the same fit config function, as nothing changes for eval
        on_evaluate_config_fn=fit_config_fn,
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        initial_parameters=initial_parameters,
        initial_control_variates=initial_control_variates,
        learning_rate=server_learning_rate,
    )

    server = ScaffoldServer(client_manager, client_model, strategy, checkpointer)

    fl.server.start_server(
        server=server,
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=config["n_server_rounds"]),
    )

    log(INFO, f"Best Aggregated (Weighted) Loss seen by the Server: \n{checkpointer.best_metric}")

    # Shutdown the server gracefully
    server.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FL Server Main")
    parser.add_argument(
        "--artifact_dir",
        action="store",
        type=str,
        help="Path to save server artifacts such as logs and model checkpoints",
        required=True,
    )
    parser.add_argument(
        "--run_name",
        action="store",
        help="Name of the run, model checkpoints will be saved under a subfolder with this name",
        required=True,
    )
    parser.add_argument(
        "--config_path",
        action="store",
        type=str,
        help="Path to configuration file.",
        default="config.yaml",
    )
    parser.add_argument(
        "--server_address",
        action="store",
        type=str,
        help="Server Address to be used to communicate with the clients",
        default="0.0.0.0:8080",
    )
    parser.add_argument(
        "--server_learning_rate",
        action="store",
        type=float,
        help="Learning rate for server side",
        required=True,
    )
    args = parser.parse_args()

    config = load_config(args.config_path)
    log(INFO, f"Server Address: {args.server_address}")
    log(INFO, f"Server Learning Rate: {args.server_learning_rate}")
    main(config, args.server_address, args.artifact_dir, args.run_name, args.server_learning_rate)
