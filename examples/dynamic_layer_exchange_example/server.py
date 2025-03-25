import argparse
from functools import partial
from typing import Any

import flwr as fl
from flwr.common.typing import Config
from flwr.server.client_manager import SimpleClientManager

from examples.models.cnn_model import Net
from fl4health.metrics.metric_aggregation import evaluate_metrics_aggregation_fn, fit_metrics_aggregation_fn
from fl4health.servers.base_server import FlServer
from fl4health.strategies.fedavg_dynamic_layer import FedAvgDynamicLayer
from fl4health.utils.config import load_config, make_dict_with_epochs_or_steps
from fl4health.utils.parameter_extraction import get_all_model_parameters


def fit_config(
    batch_size: int,
    normalize: bool,
    filter_by_percentage: bool,
    select_drift_more: bool,
    norm_threshold: float,
    exchange_percentage: float,
    sample_percentage: float,
    beta: float,
    current_server_round: int,
    local_epochs: int | None = None,
    local_steps: int | None = None,
) -> Config:
    config: Config = {
        **make_dict_with_epochs_or_steps(local_epochs, local_steps),
        "batch_size": batch_size,
        "normalize": normalize,
        "filter_by_percentage": filter_by_percentage,
        "select_drift_more": select_drift_more,
        "norm_threshold": norm_threshold,
        "exchange_percentage": exchange_percentage,
        "sample_percentage": sample_percentage,
        "beta": beta,
        "current_server_round": current_server_round,
    }
    return config


def main(config: dict[str, Any]) -> None:
    # This function will be used to produce a config that is sent to each client to initialize their own environment
    fit_config_fn = partial(
        fit_config,
        config["batch_size"],
        config["normalize"],
        config["filter_by_percentage"],
        config["select_drift_more"],
        config["norm_threshold"],
        config["exchange_percentage"],
        config["sample_percentage"],
        config["beta"],
        local_epochs=config.get("local_epochs"),
        local_steps=config.get("local_steps"),
    )

    # Initializing the model on the server side
    model = Net()

    # Server performs simple FedAveraging as its server-side optimization strategy
    strategy = FedAvgDynamicLayer(
        min_fit_clients=2,
        min_evaluate_clients=2,
        # Server waits for min_available_clients before starting FL rounds
        min_available_clients=config["n_clients"],
        on_fit_config_fn=fit_config_fn,
        # We use the same fit config function, as nothing changes for eval
        on_evaluate_config_fn=fit_config_fn,
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        initial_parameters=get_all_model_parameters(model),
    )

    client_manager = SimpleClientManager()
    server = FlServer(client_manager=client_manager, fl_config=config, strategy=strategy, accept_failures=False)

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
        default="examples/dynamic_layer_exchange_example/config.yaml",
    )
    args = parser.parse_args()

    config = load_config(args.config_path)

    main(config)
