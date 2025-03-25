import argparse
from functools import partial
from typing import Any

import flwr as fl
from flwr.common.typing import Config
from flwr.server.client_manager import SimpleClientManager

from examples.models.masked_model import Masked4Cnn
from fl4health.metrics.metric_aggregation import evaluate_metrics_aggregation_fn, fit_metrics_aggregation_fn
from fl4health.servers.fedpm_server import FedPmServer
from fl4health.strategies.fedpm import FedPm
from fl4health.utils.config import load_config, make_dict_with_epochs_or_steps
from fl4health.utils.parameter_extraction import get_all_model_parameters


def fit_config(
    batch_size: int,
    n_server_rounds: int,
    downsampling_ratio: float,
    is_masked_model: bool,
    priors_reset_frequency: int,
    current_round: int,
    local_epochs: int | None = None,
    local_steps: int | None = None,
) -> Config:
    return {
        **make_dict_with_epochs_or_steps(local_epochs, local_steps),
        "batch_size": batch_size,
        "n_server_rounds": n_server_rounds,
        "downsampling_ratio": downsampling_ratio,
        "is_masked_model": is_masked_model,
        "priors_reset_frequency": priors_reset_frequency,
        "current_server_round": current_round,
    }


def main(config: dict[str, Any]) -> None:
    # This function will be used to produce a config that is sent to each client to initialize their own environment
    fit_config_fn = partial(
        fit_config,
        config["batch_size"],
        config["n_server_rounds"],
        config["downsampling_ratio"],
        config["is_masked_model"],
        config["priors_reset_frequency"],
        local_epochs=config.get("local_epochs"),
        local_steps=config.get("local_steps"),
    )

    initial_model = Masked4Cnn()

    # Server performs simple FedPM as its server-side optimization strategy
    strategy = FedPm(
        min_fit_clients=config["n_clients"],
        min_evaluate_clients=config["n_clients"],
        # Server waits for min_available_clients before starting FL rounds
        min_available_clients=config["n_clients"],
        on_fit_config_fn=fit_config_fn,
        # We use the same fit config function, as nothing changes for eval
        on_evaluate_config_fn=fit_config_fn,
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        initial_parameters=get_all_model_parameters(initial_model),
        # Perform Bayesian aggregation.
        bayesian_aggregation=True,
        accept_failures=False,
    )

    client_manager = SimpleClientManager()
    server = FedPmServer(
        client_manager, fl_config=config, strategy=strategy, reset_frequency=config["priors_reset_frequency"]
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
        default="examples/fedpm_example/config.yaml",
    )
    args = parser.parse_args()

    config = load_config(args.config_path)

    main(config)
