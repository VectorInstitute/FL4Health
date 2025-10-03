import argparse
import os
from functools import partial
from logging import INFO
from typing import Any

import flwr as fl
from flamby.datasets.fed_isic2019 import Baseline
from flwr.common.logger import log
from flwr.server.client_manager import SimpleClientManager
from flwr.server.strategy import FedAvg

from fl4health.checkpointing.checkpointer import BestLossTorchModuleCheckpointer, LatestTorchModuleCheckpointer
from fl4health.checkpointing.server_module import BaseServerCheckpointAndStateModule
from fl4health.metrics.metric_aggregation import evaluate_metrics_aggregation_fn, fit_metrics_aggregation_fn
from fl4health.parameter_exchange.full_exchanger import FullParameterExchanger
from fl4health.utils.config import load_config
from fl4health.utils.parameter_extraction import get_all_model_parameters
from fl4health.utils.random import set_all_random_seeds
from research.flamby.flamby_servers.full_exchange_server import FullExchangeServer
from research.flamby.utils import fit_config


def main(config: dict[str, Any], server_address: str, checkpoint_stub: str, run_name: str) -> None:
    # This function will be used to produce a config that is sent to each client to initialize their own environment
    fit_config_fn = partial(
        fit_config,
        config["local_steps"],
        config["n_server_rounds"],
    )

    model = Baseline()

    checkpoint_dir = os.path.join(checkpoint_stub, run_name)
    best_checkpoint_name = "server_best_model.pkl"
    last_checkpoint_name = "server_last_model.pkl"
    checkpointers = [
        BestLossTorchModuleCheckpointer(checkpoint_dir, best_checkpoint_name),
        LatestTorchModuleCheckpointer(checkpoint_dir, last_checkpoint_name),
    ]

    checkpoint_and_state_module = BaseServerCheckpointAndStateModule(
        model=model, parameter_exchanger=FullParameterExchanger(), model_checkpointers=checkpointers
    )

    client_manager = SimpleClientManager()
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
        initial_parameters=get_all_model_parameters(model),
    )

    server = FullExchangeServer(
        client_manager=client_manager,
        fl_config=config,
        strategy=strategy,
        checkpoint_and_state_module=checkpoint_and_state_module,
    )

    fl.server.start_server(
        server=server,
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=config["n_server_rounds"]),
    )

    assert isinstance(checkpointers[0], BestLossTorchModuleCheckpointer)
    log(INFO, f"Best Aggregated (Weighted) Loss seen by the Server: \n{checkpointers[0].best_score}")

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

    main(config, args.server_address, args.artifact_dir, args.run_name)
