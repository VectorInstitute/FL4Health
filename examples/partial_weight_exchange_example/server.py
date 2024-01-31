import argparse
from functools import partial
from logging import INFO
from typing import Any, Dict

import flwr as fl
from flwr.common.logger import log
from flwr.common.typing import Config
from torchtext.models import ROBERTA_BASE_ENCODER, RobertaClassificationHead

from examples.simple_metric_aggregation import evaluate_metrics_aggregation_fn, fit_metrics_aggregation_fn
from examples.utils.functions import get_all_model_parameters
from fl4health.client_managers.poisson_sampling_manager import PoissonSamplingClientManager
from fl4health.server.base_server import FlServer
from fl4health.strategies.fedavg_dynamic_layer import FedAvgDynamicLayer
from fl4health.utils.config import load_config


def construct_config(
    current_round: int,
    local_epochs: int,
    batch_size: int,
    num_classes: int,
    sequence_length: int,
    normalize: bool,
    filter_by_percentage: bool,
    norm_threshold: float,
    exchange_percentage: float,
    sample_percentage: float,
    beta: float,
) -> Config:
    assert 0 < sample_percentage <= 1
    assert 0 < beta
    return {
        "current_server_round": current_round,
        "local_epochs": local_epochs,
        "batch_size": batch_size,
        "num_classes": num_classes,
        "sequence_length": sequence_length,
        "normalize": normalize,
        "filter_by_percentage": filter_by_percentage,
        "norm_threshold": norm_threshold,
        "exchange_percentage": exchange_percentage,
        "sample_percentage": sample_percentage,
        "beta": beta,
    }


def fit_config(
    local_epochs: int,
    batch_size: int,
    num_classes: int,
    sequence_length: int,
    normalize: bool,
    filter_by_percentage: bool,
    norm_threshold: float,
    exchange_percentage: float,
    sample_percentage: float,
    beta: float,
    current_round: int,
) -> Config:
    return construct_config(
        current_round,
        local_epochs,
        batch_size,
        num_classes,
        sequence_length,
        normalize,
        filter_by_percentage,
        norm_threshold,
        exchange_percentage,
        sample_percentage,
        beta,
    )


def eval_config(
    local_epochs: int,
    batch_size: int,
    num_classes: int,
    sequence_length: int,
    normalize: bool,
    filter_by_percentage: bool,
    norm_threshold: float,
    exchange_percentage: float,
    sample_percentage: float,
    beta: float,
    n_server_rounds: int,
    current_round: int,
) -> Config:
    config = construct_config(
        current_round,
        local_epochs,
        batch_size,
        num_classes,
        sequence_length,
        normalize,
        filter_by_percentage,
        norm_threshold,
        exchange_percentage,
        sample_percentage,
        beta,
    )
    config["testing"] = n_server_rounds == current_round
    return config


def main(config: Dict[str, Any], server_address: str) -> None:
    # This function will be used to produce a config that is sent to each client to initialize their own environment
    fit_config_fn = partial(
        fit_config,
        config["local_epochs"],
        config["batch_size"],
        config["num_classes"],
        config["sequence_length"],
        config["normalize"],
        config["filter_by_percentage"],
        config["norm_threshold"],
        config["exchange_percentage"],
        config["sample_percentage"],
        config["beta"],
    )

    eval_config_fn = partial(
        eval_config,
        config["local_epochs"],
        config["batch_size"],
        config["num_classes"],
        config["sequence_length"],
        config["normalize"],
        config["filter_by_percentage"],
        config["norm_threshold"],
        config["exchange_percentage"],
        config["sample_percentage"],
        config["beta"],
        config["n_server_rounds"],
    )

    classifier_head = RobertaClassificationHead(num_classes=config["num_classes"], input_dim=768)
    initial_model = ROBERTA_BASE_ENCODER.get_model(head=classifier_head)

    # Since clients can send different tensors to the server, we perform weighted averaging separately on each tensor.
    strategy = FedAvgDynamicLayer(
        # Server waits for min_available_clients before starting FL rounds
        min_available_clients=config["n_clients"],
        on_fit_config_fn=fit_config_fn,
        on_evaluate_config_fn=eval_config_fn,
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        initial_parameters=get_all_model_parameters(initial_model),
    )

    client_manager = PoissonSamplingClientManager()
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
