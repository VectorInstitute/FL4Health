import argparse
from functools import partial
from typing import Any

import flwr as fl
from flwr.common.typing import Config
from flwr.server.client_manager import SimpleClientManager

from examples.models.cnn_model import Net
from fl4health.metrics.metric_aggregation import evaluate_metrics_aggregation_fn, fit_metrics_aggregation_fn
from fl4health.servers.base_server import FlServer
from fl4health.strategies.fedavg_sparse_coo_tensor import FedAvgSparseCooTensor
from fl4health.utils.config import load_config, make_dict_with_epochs_or_steps
from fl4health.utils.parameter_extraction import get_all_model_parameters


def fit_config(
    batch_size: int,
    sparsity_level: float,
    current_server_round: int,
    local_epochs: int | None = None,
    local_steps: int | None = None,
) -> Config:
    config: Config = {
        **make_dict_with_epochs_or_steps(local_epochs, local_steps),
        "batch_size": batch_size,
        "sparsity_level": sparsity_level,
        "current_server_round": current_server_round,
    }
    return config


def main(config: dict[str, Any]) -> None:
    # This function will be used to produce a config that is sent to each client to initialize their own environment
    fit_config_fn = partial(
        fit_config,
        config["batch_size"],
        config["sparsity_level"],
        local_epochs=config.get("local_epochs"),
        local_steps=config.get("local_steps"),
    )

    # Initializing the model on the server side
    model = Net()

    # Server performs simple FedAveraging as its server-side optimization strategy
    strategy = FedAvgSparseCooTensor(
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
        accept_failures=False,
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
        default="examples/sparse_tensor_partial_exchange_example/config.yaml",
    )
    args = parser.parse_args()

    config = load_config(args.config_path)

    main(config)
