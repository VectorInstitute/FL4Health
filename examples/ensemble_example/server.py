import argparse
from functools import partial
from typing import Any, Dict

import flwr as fl
import torch.nn as nn
from flwr.common.parameter import ndarrays_to_parameters
from flwr.common.typing import Config, Parameters
from flwr.server.strategy import FedAvg

from examples.models.ensemble_cnn import ConfigurableMnistNet
from examples.simple_metric_aggregation import evaluate_metrics_aggregation_fn, fit_metrics_aggregation_fn
from fl4health.model_bases.ensemble_base import EnsembleModel
from fl4health.utils.config import load_config


def get_initial_model_parameters() -> Parameters:
    # Initializing the model parameters on the server side.
    # Currently uses the Pytorch default initialization for the model parameters.
    ensemble_models: Dict[str, nn.Module] = {
        "model_0": ConfigurableMnistNet(out_channel_mult=1),
        "model_1": ConfigurableMnistNet(out_channel_mult=2),
        "model_2": ConfigurableMnistNet(out_channel_mult=3),
    }
    initial_model = EnsembleModel(ensemble_models)
    return ndarrays_to_parameters([val.cpu().numpy() for _, val in initial_model.state_dict().items()])


def fit_config(
    sample_percentage: float, local_epochs: int, batch_size: int, n_server_rounds: int, current_round: int
) -> Config:
    return {
        "current_server_round": current_round,
        "local_epochs": local_epochs,
        "batch_size": batch_size,
        "n_server_rounds": n_server_rounds,
        "sample_percentage": sample_percentage,
    }


def main(config: Dict[str, Any]) -> None:
    # This function will be used to produce a config that is sent to each client to initialize their own environment
    fit_config_fn = partial(
        fit_config,
        float(config["sample_percentage"]),
        config["local_epochs"],
        config["batch_size"],
        config["n_server_rounds"],
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
        default="examples/ensemble_example/config.yaml",
    )
    args = parser.parse_args()

    config = load_config(args.config_path)

    main(config)