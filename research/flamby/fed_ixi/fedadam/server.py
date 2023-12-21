import argparse
import os
from functools import partial
from logging import INFO
from typing import Any, Dict

import flwr as fl
from flwr.common.logger import log
from flwr.server.client_manager import SimpleClientManager
from flwr.server.strategy import FedAdam

from fl4health.checkpointing.checkpointer import BestMetricTorchCheckpointer, LatestTorchCheckpointer
from fl4health.utils.config import load_config
from research.flamby.fed_ixi.fedadam.fedadam_model import FedAdamUNet
from research.flamby.flamby_servers.full_exchange_server import FullExchangeServer
from research.flamby.utils import (
    evaluate_metrics_aggregation_fn,
    fit_config,
    fit_metrics_aggregation_fn,
    get_initial_model_parameters,
    summarize_model_info,
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
    federated_checkpointing: bool = config.get("federated_checkpointing", True)
    log(INFO, f"Performing Federated Checkpointing: {federated_checkpointing}")
    checkpointer = (
        BestMetricTorchCheckpointer(checkpoint_dir, checkpoint_name)
        if federated_checkpointing
        else LatestTorchCheckpointer(checkpoint_dir, checkpoint_name)
    )

    client_manager = SimpleClientManager()
    model = FedAdamUNet()
    summarize_model_info(model)

    strategy = FedAdam(
        min_fit_clients=config["n_clients"],
        min_evaluate_clients=config["n_clients"],
        # Server waits for min_available_clients before starting FL rounds
        min_available_clients=config["n_clients"],
        on_fit_config_fn=fit_config_fn,
        # We use the same fit config function, as nothing changes for eval
        on_evaluate_config_fn=fit_config_fn,
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        initial_parameters=get_initial_model_parameters(model),
        eta=server_learning_rate,
    )

    server = FullExchangeServer(client_manager, model, strategy, checkpointer=checkpointer)

    fl.server.start_server(
        server=server,
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=config["n_server_rounds"]),
    )

    if federated_checkpointing:
        assert isinstance(checkpointer, BestMetricTorchCheckpointer)
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
