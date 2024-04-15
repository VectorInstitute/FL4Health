import argparse
from functools import partial
from logging import INFO
from typing import Any, Dict

import flwr as fl
from flwr.common.logger import log
from flwr.common.typing import Config
from flwr.server.client_manager import SimpleClientManager
from transformers import BertForSequenceClassification

from fl4health.server.base_server import FlServer
from fl4health.strategies.fedavg_sparse_coo_tensor import FedAvgSparseCooTensor
from fl4health.utils.config import load_config
from fl4health.utils.functions import get_all_model_parameters
from fl4health.utils.metric_aggregation import evaluate_metrics_aggregation_fn, fit_metrics_aggregation_fn


def construct_config(
    current_round: int,
    local_steps: int,
    batch_size: int,
    num_classes: int,
    sample_percentage: float,
    beta: float,
) -> Config:
    assert 0 < sample_percentage <= 1
    assert 0 < beta
    return {
        "current_server_round": current_round,
        "local_steps": local_steps,
        "batch_size": batch_size,
        "num_classes": num_classes,
        "sample_percentage": sample_percentage,
        "beta": beta,
    }


def fit_config(
    local_steps: int,
    batch_size: int,
    num_classes: int,
    sample_percentage: float,
    beta: float,
    current_round: int,
) -> Config:
    return construct_config(
        current_round,
        local_steps,
        batch_size,
        num_classes,
        sample_percentage,
        beta,
    )


def main(config: Dict[str, Any], server_address: str) -> None:
    # This function will be used to produce a config that is sent to each client to initialize their own environment
    fit_config_fn = partial(
        fit_config,
        config["local_steps"],
        config["batch_size"],
        config["num_classes"],
        config["sample_percentage"],
        config["beta"],
    )

    initial_model = BertForSequenceClassification.from_pretrained(
        "google-bert/bert-base-cased", num_labels=config["num_classes"]
    )

    # Since clients can send different tensors to the server, we perform weighted averaging separately on each tensor.
    strategy = FedAvgSparseCooTensor(
        # Server waits for min_available_clients before starting FL rounds
        min_available_clients=config["n_clients"],
        on_fit_config_fn=fit_config_fn,
        on_evaluate_config_fn=fit_config_fn,
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        initial_parameters=get_all_model_parameters(initial_model),
    )

    client_manager = SimpleClientManager()
    server = FlServer(client_manager, strategy)

    # grpc_max_message_length is reset here so the entire model can be exchanged between the server and clients.
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
