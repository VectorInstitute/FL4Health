import argparse
from typing import Any, Dict, List, Tuple

import flwr as fl
from flwr.common.parameter import ndarrays_to_parameters
from flwr.common.typing import Config, Metrics, Parameters

from examples.models.logistic_regression import LogisticRegression
from examples.simple_metric_aggregation import metric_aggregation, normalize_metrics
from fl4health.client_managers.poisson_sampling_manager import PoissonSamplingClientManager
from fl4health.server.tabular_feature_alignment_server import TabularFeatureAlignmentServer
from fl4health.strategies.basic_fedavg import BasicFedAvg
from fl4health.utils.config import load_config


def get_initial_model_parameters(input_dim: int, output_dim: int) -> Parameters:
    # FedAdam requires that we provide server side parameter initialization.
    # Currently uses the Pytorch default initialization for the model parameters.
    initial_model = LogisticRegression(input_dim, output_dim)
    return ndarrays_to_parameters([val.cpu().numpy() for _, val in initial_model.state_dict().items()])


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


def construct_config(_: int, local_epochs: int, batch_size: int, id_column: str, target_column: str) -> Config:
    # NOTE: The omitted variable is server_round which allows for dynamically changing the config each round
    return {
        "local_epochs": local_epochs,
        "batch_size": batch_size,
        "id_column": id_column,
        "target_column": target_column,
    }


def fit_config(
    local_epochs: int,
    batch_size: int,
    id_column: str,
    target_column: str,
    current_round: int,
) -> Config:
    config = construct_config(current_round, local_epochs, batch_size, id_column, target_column)
    config["format_specified"] = current_round != 1
    return config


def main(config: Dict[str, Any]) -> None:
    client_manager = PoissonSamplingClientManager()
    # Server performs simple FedAveraging as its server-side optimization strategy
    strategy = BasicFedAvg(
        min_fit_clients=config["n_clients"],
        min_evaluate_clients=config["n_clients"],
        # Server waits for min_available_clients before starting FL rounds
        min_available_clients=config["n_clients"],
        on_fit_config_fn=None,
        on_evaluate_config_fn=None,
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        initial_parameters=None,
    )

    server = TabularFeatureAlignmentServer(
        client_manager=client_manager,
        config=config,
        initialize_parameters=get_initial_model_parameters,
        strategy=strategy,
    )

    fl.server.start_server(
        server=server,
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=config["n_server_rounds"]),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FL Server Main")
    parser.add_argument(
        "--config_path",
        action="store",
        type=str,
        help="Path to configuration file.",
        default="examples/feature_alignment_example/config.yaml",
    )
    args = parser.parse_args()

    config = load_config(args.config_path)

    main(config)
