import argparse
from functools import partial
from logging import INFO
from typing import Any, Dict, List, Tuple

import flwr as fl
import numpy as np
from flwr.common.logger import log
from flwr.common.parameter import ndarrays_to_parameters
from flwr.common.typing import Config, Metrics, Parameters
from torchtext.models import ROBERTA_BASE_ENCODER, RobertaClassificationHead

from examples.simple_metric_aggregation import metric_aggregation, normalize_metrics
from fl4health.client_managers.poisson_sampling_manager import PoissonSamplingClientManager
from fl4health.strategies.fedavg_dynamic_layer import FedAvgDynamicLayer
from fl4health.utils.config import load_config


def fit_metrics_aggregation_fn(all_client_metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # This function is run by the server to aggregate metrics returned by each clients fit function
    # NOTE: The first value of the tuple is number of examples for FedAvg
    total_examples, aggregated_metrics = metric_aggregation(all_client_metrics)
    return normalize_metrics(total_examples, aggregated_metrics)


def evaluate_metrics_aggregation_fn(all_client_metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # This function is run by the server to aggregate metrics returned by each clients evaluate function
    # NOTE: The first value of the tuple is number of examples for FedAvg
    total_examples, aggregated_metrics = metric_aggregation(all_client_metrics)
    return normalize_metrics(total_examples, aggregated_metrics)


def get_initial_model_parameters(num_classes: int) -> Parameters:
    # Initializing the model parameters on the server side.
    classifier_head = RobertaClassificationHead(num_classes=num_classes, input_dim=768)
    initial_model = ROBERTA_BASE_ENCODER.get_model(head=classifier_head)
    names = []
    params = []
    for key, val in initial_model.state_dict().items():
        names.append(key)
        params.append(val.cpu().numpy())
    return ndarrays_to_parameters(params + [np.array(names)])


def construct_fit_config(
    _: int,
    local_epochs: int,
    batch_size: int,
    num_classes: int,
    sequence_length: int,
    normalize: bool,
    filter_by_percentage: bool,
    sample_percentage: float,
    beta: float,
) -> Config:
    assert 0 < sample_percentage <= 1
    assert 0 < beta
    return {
        "local_epochs": local_epochs,
        "batch_size": batch_size,
        "num_classes": num_classes,
        "sequence_length": sequence_length,
        "normalize": normalize,
        "filter_by_percentage": filter_by_percentage,
        "sample_percentage": sample_percentage,
        "beta": beta,
    }


def construct_eval_config(
    _: int,
    num_classes: int,
) -> Config:
    return {"num_classes": num_classes}


def fit_config(
    local_epochs: int,
    batch_size: int,
    num_classes: int,
    sequence_length: int,
    normalize: bool,
    filter_by_percentage: bool,
    sample_percentage: float,
    beta: float,
    current_round: int,
) -> Config:
    return construct_fit_config(
        current_round,
        local_epochs,
        batch_size,
        num_classes,
        sequence_length,
        normalize,
        filter_by_percentage,
        sample_percentage,
        beta,
    )


def eval_config(
    num_classes: int,
    n_server_rounds: int,
    current_round: int,
) -> Config:
    config = construct_eval_config(current_round, num_classes)
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
        config["sample_percentage"],
        config["beta"],
    )

    eval_config_fn = partial(
        eval_config,
        config["num_classes"],
        config["n_server_rounds"],
    )

    # Server performs simple FedAveraging as its server-side optimization strategy
    strategy = FedAvgDynamicLayer(
        # Server waits for min_available_clients before starting FL rounds
        min_available_clients=config["n_clients"],
        on_fit_config_fn=fit_config_fn,
        on_evaluate_config_fn=eval_config_fn,
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        initial_parameters=get_initial_model_parameters(config["num_classes"]),
    )

    client_manager = PoissonSamplingClientManager()

    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=config["n_server_rounds"]),
        strategy=strategy,
        client_manager=client_manager,
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
