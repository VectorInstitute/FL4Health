import argparse
from functools import partial
from logging import INFO
from typing import Any

import flwr as fl
from flwr.common.logger import log
from flwr.common.typing import Config
from flwr.server.client_manager import SimpleClientManager
from flwr.server.strategy import FedAvg
from transformers import BertForSequenceClassification

from fl4health.checkpointing.checkpointer import BestLossTorchModuleCheckpointer, LatestTorchModuleCheckpointer
from fl4health.checkpointing.server_module import BaseServerCheckpointAndStateModule
from fl4health.metrics.metric_aggregation import evaluate_metrics_aggregation_fn, fit_metrics_aggregation_fn
from fl4health.parameter_exchange.full_exchanger import FullParameterExchanger
from fl4health.servers.base_server import FlServer
from fl4health.utils.config import load_config, make_dict_with_epochs_or_steps
from fl4health.utils.parameter_extraction import get_all_model_parameters


def fit_config(
    batch_size: int,
    num_classes: int,
    n_server_rounds: int,
    sample_percentage: float,
    beta: float,
    current_round: int,
    local_epochs: int | None = None,
    local_steps: int | None = None,
) -> Config:
    assert 0 < sample_percentage <= 1
    assert beta > 0
    return {
        **make_dict_with_epochs_or_steps(local_epochs, local_steps),
        "batch_size": batch_size,
        "num_classes": num_classes,
        "n_server_rounds": n_server_rounds,
        "sample_percentage": sample_percentage,
        "beta": beta,
        "current_server_round": current_round,
    }


def main(config: dict[str, Any], server_address: str) -> None:
    # This function will be used to produce a config that is sent to each client to initialize their own environment
    fit_config_fn = partial(
        fit_config,
        config["batch_size"],
        config["num_classes"],
        config["n_server_rounds"],
        config["sample_percentage"],
        config["beta"],
        local_epochs=config.get("local_epochs"),
        local_steps=config.get("local_steps"),
    )

    # Initializing the model on the server side
    initial_model = BertForSequenceClassification.from_pretrained(
        "google-bert/bert-base-cased", num_labels=config["num_classes"]
    )
    # To facilitate checkpointing
    parameter_exchanger = FullParameterExchanger()
    checkpointers = [
        BestLossTorchModuleCheckpointer(config["checkpoint_path"], "best_model.pkl"),
        LatestTorchModuleCheckpointer(config["checkpoint_path"], "latest_model.pkl"),
    ]
    checkpoint_and_state_module = BaseServerCheckpointAndStateModule(
        model=initial_model, parameter_exchanger=parameter_exchanger, model_checkpointers=checkpointers
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
        initial_parameters=get_all_model_parameters(initial_model),
    )

    server = FlServer(
        client_manager=SimpleClientManager(),
        fl_config=config,
        strategy=strategy,
        checkpoint_and_state_module=checkpoint_and_state_module,
        accept_failures=False,
    )

    # grpc_max_message_length is reset here so the entire model can be exchanged between the server and clients.
    # Note that the server and clients must be started with the same grpc_max_message_length. Otherwise communication
    # of larger messages would still be blocked.
    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=config["n_server_rounds"]),
        server=server,
        grpc_max_message_length=1600000000,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FL Server Main")
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
