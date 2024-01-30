import argparse
from functools import partial
from typing import Any, Dict, Optional

import flwr as fl
import torch.nn as nn
from flwr.common.parameter import ndarrays_to_parameters
from flwr.common.typing import Config, Parameters
from flwr.server.client_manager import SimpleClientManager

from examples.models.cnn_model import Net
from examples.simple_metric_aggregation import evaluate_metrics_aggregation_fn, fit_metrics_aggregation_fn
from examples.utils.functions import make_dict_with_epochs_or_steps
from fl4health.server.base_server import FlServer
from fl4health.strategies.fedavg_sparse_coo_tensor import FedAvgSparseCooTensor
from fl4health.utils.config import load_config


def get_initial_model_parameters(model: nn.Module) -> Parameters:
    # Initializing the model parameters on the server side.
    return ndarrays_to_parameters([val.cpu().numpy() for _, val in model.state_dict().items()])


def fit_config(
    batch_size: int,
    sparsity_level: float,
    current_server_round: int,
    local_epochs: Optional[int] = None,
    local_steps: Optional[int] = None,
) -> Config:
    config: Config = {
        **make_dict_with_epochs_or_steps(local_epochs, local_steps),
        "batch_size": batch_size,
        "sparsity_level": sparsity_level,
        "current_server_round": current_server_round,
    }
    config["fl_fit_round"] = True
    return config


def eval_config(
    batch_size: int,
    sparsity_level: float,
    current_server_round: int,
    local_epochs: Optional[int] = None,
    local_steps: Optional[int] = None,
) -> Config:
    config: Config = {
        **make_dict_with_epochs_or_steps(local_epochs, local_steps),
        "batch_size": batch_size,
        "sparsity_level": sparsity_level,
        "current_server_round": current_server_round,
    }
    config["fl_fit_round"] = False
    return config


def main(config: Dict[str, Any]) -> None:
    # This function will be used to produce a config that is sent to each client to initialize their own environment
    fit_config_fn = partial(
        fit_config,
        config["batch_size"],
        config["sparsity_level"],
        local_epochs=config.get("local_epochs"),
        local_steps=config.get("local_steps"),
    )

    eval_config_fn = partial(
        eval_config,
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
        on_evaluate_config_fn=eval_config_fn,
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        initial_parameters=get_initial_model_parameters(model),
    )

    client_manager = SimpleClientManager()
    server = FlServer(client_manager, strategy)

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
