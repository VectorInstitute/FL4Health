import argparse
from functools import partial
from typing import Any, Dict

import flwr as fl
import torch.nn as nn
from flwr.common.parameter import ndarrays_to_parameters
from flwr.common.typing import Config, Parameters

from examples.models.cnn_model import MnistNetWithBnAndFrozen
from examples.simple_metric_aggregation import evaluate_metrics_aggregation_fn, fit_metrics_aggregation_fn
from fl4health.client_managers.poisson_sampling_manager import PoissonSamplingClientManager
from fl4health.server.scaffold_server import ScaffoldServer
from fl4health.strategies.scaffold import Scaffold
from fl4health.utils.config import load_config


def get_initial_model_parameters(initial_model: nn.Module) -> Parameters:
    # Initializing the model parameters on the server side.
    model_weights = [val.cpu().numpy() for _, val in initial_model.state_dict().items()]
    return ndarrays_to_parameters(model_weights)


def fit_config(local_steps: int, batch_size: int, n_server_rounds: int, current_round: int) -> Config:
    return {
        "local_steps": local_steps,
        "batch_size": batch_size,
        "n_server_rounds": n_server_rounds,
        "current_server_round": current_round,
    }


def main(config: Dict[str, Any]) -> None:
    # This function will be used to produce a config that is sent to each client to initialize their own environment
    fit_config_fn = partial(
        fit_config,
        config["local_steps"],
        config["batch_size"],
        config["n_server_rounds"],
    )

    model = MnistNetWithBnAndFrozen()

    # Initialize Scaffold strategy to handle aggregation of weights and corresponding control variates
    strategy = Scaffold(
        min_available_clients=config["n_clients"],
        on_fit_config_fn=fit_config_fn,
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        # We use the same fit config function, as nothing changes for eval
        on_evaluate_config_fn=fit_config_fn,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        initial_parameters=get_initial_model_parameters(model),
        model=model,
    )

    # ClientManager that performs Poisson type sampling
    client_manager = PoissonSamplingClientManager()

    server = ScaffoldServer(client_manager=client_manager, strategy=strategy, warm_start=True)
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
        default="examples/scaffold_example/config.yaml",
    )
    args = parser.parse_args()

    config = load_config(args.config_path)

    main(config)
