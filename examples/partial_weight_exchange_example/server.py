import argparse
from functools import partial
from logging import INFO
from typing import Any, Dict, List, Tuple

import flwr as fl
import numpy as np
from flwr.common.logger import log
from flwr.common.parameter import ndarrays_to_parameters
from flwr.common.typing import Config, Metrics, Parameters
from torchtext.models import XLMR_BASE_ENCODER, RobertaClassificationHead

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


def get_initial_model_parameters(num_classes: int, input_dimension: int) -> Parameters:
    # Initializing the model parameters on the server side.
    classifier_head = RobertaClassificationHead(num_classes=num_classes, input_dim=input_dimension)
    initial_model = XLMR_BASE_ENCODER.get_model(head=classifier_head)
    names = []
    params = []
    for key, val in initial_model.state_dict().items():
        names.append(key)
        params.append(val.cpu().numpy())
    return ndarrays_to_parameters(params + [np.array(names)])


def fit_config(
    local_epochs: int,
    batch_size: int,
    current_round: int,
) -> Config:
    return {"local_epochs": local_epochs, "batch_size": batch_size, "current_round": current_round}


def main(config: Dict[str, Any], server_address: str) -> None:
    # This function will be used to produce a config that is sent to each client to initialize their own environment
    fit_config_fn = partial(
        fit_config,
        config["local_epochs"],
        config["batch_size"],
    )

    # Server performs simple FedAveraging as its server-side optimization strategy
    strategy = FedAvgDynamicLayer(
        # min_fit_clients=config["n_clients"],
        # min_evaluate_clients=config["n_clients"],
        # Server waits for min_available_clients before starting FL rounds
        min_available_clients=config["n_clients"],
        on_fit_config_fn=fit_config_fn,
        # We use the same fit config function, as nothing changes for eval
        on_evaluate_config_fn=fit_config_fn,
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        initial_parameters=get_initial_model_parameters(config["num_classes"], config["input_dimension"]),
    )

    client_manager = PoissonSamplingClientManager()

    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=config["n_server_rounds"]),
        strategy=strategy,
        client_manager=client_manager,
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
