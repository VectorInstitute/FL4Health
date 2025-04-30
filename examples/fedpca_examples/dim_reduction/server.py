import argparse
from functools import partial
from typing import Any

import flwr as fl
from flwr.common.typing import Config
from flwr.server.client_manager import SimpleClientManager
from flwr.server.strategy import FedAvg

from examples.models.mnist_model import MnistNet
from fl4health.checkpointing.checkpointer import BestLossTorchModuleCheckpointer
from fl4health.checkpointing.server_module import BaseServerCheckpointAndStateModule
from fl4health.metrics.metric_aggregation import evaluate_metrics_aggregation_fn, fit_metrics_aggregation_fn
from fl4health.parameter_exchange.full_exchanger import FullParameterExchanger
from fl4health.servers.base_server import FlServer
from fl4health.utils.config import load_config
from fl4health.utils.parameter_extraction import get_all_model_parameters


def fit_config(
    local_epochs: int,
    batch_size: int,
    new_dimension: int,
    pca_path: str,
    current_server_round: int,
) -> Config:
    return {
        "local_epochs": local_epochs,
        "batch_size": batch_size,
        "new_dimension": new_dimension,
        "pca_path": pca_path,
        "current_server_round": current_server_round,
    }


def main(config: dict[str, Any]) -> None:
    # This function will be used to produce a config that is sent to each client to initialize their own environment
    fit_config_fn = partial(
        fit_config,
        config["local_epochs"],
        config["batch_size"],
        config["new_dimension"],
        config["pca_path"],
    )

    # Initializing the model on the server side
    model = MnistNet(input_dim=config["new_dimension"])
    parameter_exchanger = FullParameterExchanger()

    # To facilitate checkpointing
    checkpointer = BestLossTorchModuleCheckpointer(config["checkpoint_path"], "best_model.pkl")
    checkpoint_and_state_module = BaseServerCheckpointAndStateModule(
        model=model, parameter_exchanger=parameter_exchanger, model_checkpointers=checkpointer
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
        initial_parameters=get_all_model_parameters(model),
    )

    server = FlServer(
        client_manager=SimpleClientManager(),
        fl_config=config,
        strategy=strategy,
        checkpoint_and_state_module=checkpoint_and_state_module,
        accept_failures=False,
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
        default="examples/basic_example/config.yaml",
    )
    args = parser.parse_args()

    config = load_config(args.config_path)

    main(config)
