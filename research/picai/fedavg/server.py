import argparse
from functools import partial
from logging import INFO
from typing import Any, Dict

import torch
import flwr as fl
from flwr.common.logger import log
from flwr.server.client_manager import SimpleClientManager
from flwr.server.strategy import FedAvg

from fl4health.utils.config import load_config
from fl4health.parameter_exchange.full_exchanger import FullParameterExchanger
from fl4health.checkpointing.checkpointer import ServerPerEpochCheckpointer

from research.picai.picai_server import PicaiServer
from research.picai.model_utils import get_model
from research.picai.fl_utils import (
    evaluate_metrics_aggregation_fn,
    fit_config,
    fit_metrics_aggregation_fn,
    get_initial_model_parameters,
)


def main(config: Dict[str, Any], server_address: str) -> None:
    # This function will be used to produce a config that is sent to each client to initialize their own environment
    fit_config_fn = partial(
        fit_config,
        config["fold_id"],
        config["batch_size"],
        config["local_epochs"],
        config["n_server_rounds"],
    )

    client_manager = SimpleClientManager()
    model = get_model(device=torch.device("gpu"))

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
        initial_parameters=get_initial_model_parameters(model),
    )

    per_epoch_checkpointer = ServerPerEpochCheckpointer(args.artifact_dir, "server.pt")
    server = PicaiServer(
        client_manager=client_manager,
        model=model,
        parameter_exchanger=FullParameterExchanger(),
        strategy=strategy,
        per_epoch_checkpointer=per_epoch_checkpointer
    )

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
        "--artifact_dir",
        action="store",
        type=str,
        help="Path to dir to store run artifacts",
        required=True
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
    args = parser.parse_args()

    config = load_config(args.config_path)
    log(INFO, f"Server Address: {args.server_address}")
    main(config, args.server_address)
