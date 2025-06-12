import argparse
from functools import partial
from pathlib import Path
from typing import Any

import flwr as fl
from flwr.common.typing import Config
from flwr.server.client_manager import SimpleClientManager
from flwr.server.strategy import FedAvg

from examples.models.cnn_model import Net
from fl4health.checkpointing.checkpointer import BestLossTorchModuleCheckpointer, LatestTorchModuleCheckpointer
from fl4health.checkpointing.server_module import BaseServerCheckpointAndStateModule
from fl4health.checkpointing.state_checkpointer import ServerStateCheckpointer
from fl4health.metrics.metric_aggregation import evaluate_metrics_aggregation_fn, fit_metrics_aggregation_fn
from fl4health.parameter_exchange.full_exchanger import FullParameterExchanger
from fl4health.reporting import JsonReporter
from fl4health.servers.base_server import FlServer
from fl4health.utils.config import load_config, make_dict_with_epochs_or_steps
from fl4health.utils.parameter_extraction import get_all_model_parameters
from fl4health.utils.random import set_all_random_seeds


def fit_config(
    batch_size: int,
    current_server_round: int,
    local_epochs: int | None = None,
    local_steps: int | None = None,
) -> Config:
    return {
        **make_dict_with_epochs_or_steps(local_epochs, local_steps),
        "batch_size": batch_size,
        "current_server_round": current_server_round,
    }


def main(config: dict[str, Any], intermediate_server_state_dir: str, server_name: str) -> None:
    # This function will be used to produce a config that is sent to each client to initialize their own environment
    fit_config_fn = partial(
        fit_config,
        config["batch_size"],
        local_epochs=config.get("local_epochs"),
        local_steps=config.get("local_steps"),
    )

    # Initializing the model on the server side
    model = Net()
    # To facilitate checkpointing
    parameter_exchanger = FullParameterExchanger()
    checkpointers = [
        BestLossTorchModuleCheckpointer(config["checkpoint_path"], "best_model.pkl"),
        LatestTorchModuleCheckpointer(config["checkpoint_path"], "latest_model.pkl"),
    ]
    state_checkpointer = ServerStateCheckpointer(Path(intermediate_server_state_dir))
    checkpoint_and_state_module = BaseServerCheckpointAndStateModule(
        model=model,
        parameter_exchanger=parameter_exchanger,
        model_checkpointers=checkpointers,
        state_checkpointer=state_checkpointer,
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
        initial_parameters=get_all_model_parameters(model),
    )

    server = FlServer(
        client_manager=SimpleClientManager(),
        fl_config=config,
        strategy=strategy,
        reporters=[JsonReporter()],
        checkpoint_and_state_module=checkpoint_and_state_module,
        server_name=server_name,
    )

    fl.server.start_server(
        server=server,
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=config["n_server_rounds"]),
    )
    # disconnect all clients
    server.disconnect_all_clients(timeout=10)
    server.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FL Server Main")
    parser.add_argument(
        "--config_path",
        action="store",
        type=str,
        help="Path to configuration file.",
        default="tests/smoke_tests/load_from_checkpoint_example/config.yaml",
    )
    parser.add_argument(
        "--intermediate_server_state_dir",
        action="store",
        type=str,
        help="Path to intermediate checkpoint directory.",
        default="./",
    )
    parser.add_argument(
        "--server_name",
        action="store",
        type=str,
        help="Unique name to identify server.",
    )
    parser.add_argument(
        "--seed",
        action="store",
        type=int,
        help="Seed for the random number generators across python, torch, and numpy",
    )
    args = parser.parse_args()

    config = load_config(args.config_path)

    # Set the random seed for reproducibility
    set_all_random_seeds(args.seed)

    main(config, args.intermediate_server_state_dir, args.server_name)
