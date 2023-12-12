import argparse
from functools import partial
from typing import Any, Dict

import flwr as fl
from flwr.common.typing import Config, Parameters
from flwr.server.client_manager import SimpleClientManager
from flwr.server.server import Server

from examples.simple_metric_aggregation import evaluate_metrics_aggregation_fn, fit_metrics_aggregation_fn
from fl4health.strategies.fedpca import FedPCA
from fl4health.utils.config import load_config


def get_initial_model_parameters() -> Parameters:
    # Initializing the model parameters on the server side. This is not used for computing federated PCA.
    return Parameters(tensors=[], tensor_type="numpy.ndarray")


def fit_config(
    batch_size: int,
    low_rank: bool,
    full_svd: bool,
    rank_estimation: int,
    center_data: bool,
    num_components: int,
    current_server_round: int,
) -> Config:
    return {
        "batch_size": batch_size,
        "low_rank": low_rank,
        "full_svd": full_svd,
        "rank_estimation": rank_estimation,
        "center_data": center_data,
        "num_components": num_components,
        "current_server_round": current_server_round,
    }


def main(config: Dict[str, Any]) -> None:
    # This function will be used to produce a config that is sent to each client to initialize their own environment
    fit_config_fn = partial(
        fit_config,
        config["batch_size"],
        config["low_rank"],
        config["full_svd"],
        config["rank_estimation"],
        config["center_data"],
        config["num_components"],
    )

    # Initialize FedPCA strategy.
    strategy = FedPCA(
        # Server waits for min_available_clients before starting FL rounds
        min_available_clients=config["n_clients"],
        on_fit_config_fn=fit_config_fn,
        # We use the same fit config function, as nothing changes for eval
        on_evaluate_config_fn=fit_config_fn,
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        initial_parameters=get_initial_model_parameters(),
    )

    # We use the default flwr server here.
    server = Server(client_manager=SimpleClientManager(), strategy=strategy)

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
        default="examples/fedpca_examples/perform_pca/config.yaml",
    )
    args = parser.parse_args()

    config = load_config(args.config_path)

    main(config)
