import argparse
import os
from functools import partial
from logging import INFO
from typing import Any

import flwr as fl
from delirium_models.moon_model import DeliriumMoonModel
from flwr.common.logger import log
from flwr.common.parameter import ndarrays_to_parameters
from flwr.common.typing import Config, Parameters
from flwr.server.client_manager import SimpleClientManager
from flwr.server.strategy import FedAvg

# model
from mortality_models.moon_model import MortalityMoonModel
from servers.full_exchange_server import FullExchangeServer
from torch import nn
from utils.random import set_all_random_seeds

from fl4health.checkpointing.checkpointer import BestMetricTorchCheckpointer, LatestTorchModuleCheckpointer
from fl4health.utils.config import load_config
from research.gemini.simple_metric_aggregation import evaluate_metrics_aggregation_fn, fit_metrics_aggregation_fn


def get_initial_model_parameters(client_model: nn.Module) -> Parameters:
    # Initializing the model parameters on the server side.
    return ndarrays_to_parameters([val.cpu().numpy() for _, val in client_model.state_dict().items()])


def fit_config(
    local_epochs: int,
    batch_size: int,
    n_server_rounds: int,
    current_round: int,
) -> Config:
    return {
        "local_epochs": local_epochs,
        "batch_size": batch_size,
        "n_server_rounds": n_server_rounds,
        "current_server_round": current_round,
    }


def main(config: dict[str, Any], server_address: str, checkpoint_stub: str, run_name: str) -> None:
    # This function will be used to produce a config that is sent to each client to initialize their own environment
    fit_config_fn = partial(
        fit_config,
        config["local_epochs"],
        config["batch_size"],
        config["n_server_rounds"],
    )

    checkpoint_dir = os.path.join(checkpoint_stub, run_name)
    checkpoint_name = "server_best_model.pkl"
    federated_checkpointing: bool = config.get("federated_checkpointing", True)
    log(INFO, f"Performing Federated Checkpointing: {federated_checkpointing}")
    checkpointer = (
        BestMetricTorchCheckpointer(checkpoint_dir, checkpoint_name)
        if federated_checkpointing
        else LatestTorchModuleCheckpointer(checkpoint_dir, checkpoint_name)
    )

    client_manager = SimpleClientManager()
    if int(config["n_clients"]) == 6:
        # delirium
        client_model = DeliriumMoonModel(input_dim=8093, output_dim=1)
    else:
        # mortality
        client_model = MortalityMoonModel(input_dim=35, output_dim=1)

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
        initial_parameters=get_initial_model_parameters(client_model),
    )

    server = FullExchangeServer(client_manager, client_model, strategy, checkpointer=checkpointer)

    fl.server.start_server(
        server=server,
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=config["n_server_rounds"]),
    )

    if federated_checkpointing:
        assert isinstance(checkpointer, BestMetricTorchCheckpointer)
        log(
            INFO,
            f"Best Aggregated (Weighted) Loss seen by the Server: \n{checkpointer.best_metric}",
        )

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
        default="moon/config.yaml",
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
    # Set the random seed for reproducibility
    set_all_random_seeds(args.seed)
    config = load_config(args.config_path)
    log(INFO, f"Server Address: {args.server_address}")
    main(config, args.server_address, args.artifact_dir, args.run_name)
