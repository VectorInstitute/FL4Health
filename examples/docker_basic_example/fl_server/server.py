import argparse
from functools import partial
from typing import Any, Dict, List, Tuple

import flwr as fl
from flwr.common.parameter import ndarrays_to_parameters
from flwr.common.typing import Config, Metrics, Parameters
from flwr.server.strategy import FedAvg

from examples.docker_basic_example.model import Net
from fl4health.utils.config import load_config


def metric_aggregation(all_client_metrics: List[Tuple[int, Metrics]]) -> Tuple[int, Metrics]:
    aggregated_metrics: Metrics = {}
    total_examples = 0
    # Run through all of the metrics
    for num_examples_on_client, client_metrics in all_client_metrics:
        total_examples += num_examples_on_client
        for metric_name, metric_value in client_metrics.items():
            # Here we assume each metric is normalized by the number of examples on the client. So we scale up to
            # get the "raw" value
            if metric_name in aggregated_metrics:
                aggregated_metrics[metric_name] += num_examples_on_client * metric_value
            else:
                aggregated_metrics[metric_name] = num_examples_on_client * metric_value
    return total_examples, aggregated_metrics


def normalize_metrics(total_examples: int, aggregated_metrics: Metrics) -> Metrics:
    # Normalize all metric values by the total count of examples seen.
    return {metric_name: metric_value / total_examples for metric_name, metric_value in aggregated_metrics.items()}


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


def fit_config(
    local_epochs: int,
    batch_size: int,
    n_server_rounds: int,
) -> Config:
    return {"local_epochs": local_epochs, "batch_size": batch_size, "n_server_rounds": n_server_rounds}


def get_initial_model_parameters() -> Parameters:
    # FedAdam requires that we provide server side parameter initialization.
    # Currently uses the Pytorch default initialization for the model parameters.
    initial_model = Net()
    return ndarrays_to_parameters([val.cpu().numpy() for _, val in initial_model.state_dict().items()])


def main(config: Dict[str, Any]) -> None:
    # This function will be used to produce a config that is sent to each client to initialize their own environment
    fit_config_fn = partial(
        fit_config,
        config["local_epochs"],
        config["batch_size"],
    )

    # Server performs simple FedAveraging as it's server-side optimization strategy
    strategy = FedAvg(
        min_fit_clients=config["n_clients"],
        min_evaluate_clients=config["n_clients"],
        # Server waits for min_available_clients before starting FL rounds
        min_available_clients=config["n_clients"],
        on_fit_config_fn=fit_config_fn,
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        initial_parameters=get_initial_model_parameters(),
    )

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=config["n_server_rounds"]),
        strategy=strategy,
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
    args = parser.parse_args()

    config = load_config(args.config_path)

    main(config)
