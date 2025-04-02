import argparse
from functools import partial
from typing import Any

import flwr as fl
from flwr.common.typing import Config
from flwr.server.client_manager import SimpleClientManager
from flwr.server.strategy import FedAvg

from examples.models.sequential_split_models import (
    SequentialGlobalFeatureExtractorCifar,
    SequentialLocalPredictionHeadCifar,
)
from fl4health.metrics.metric_aggregation import evaluate_metrics_aggregation_fn, fit_metrics_aggregation_fn
from fl4health.model_bases.fedrep_base import FedRepModel
from fl4health.servers.base_server import FlServer
from fl4health.utils.config import load_config
from fl4health.utils.parameter_extraction import get_all_model_parameters


def fit_config(
    batch_size: int,
    n_server_rounds: int,
    sample_percentage: float,
    beta: float,
    local_head_steps: int,
    local_representation_steps: int,
    current_round: int,
) -> Config:
    return {
        "batch_size": batch_size,
        "n_server_rounds": n_server_rounds,
        "current_server_round": current_round,
        "sample_percentage": sample_percentage,
        "beta": beta,
        "local_head_steps": local_head_steps,
        "local_rep_steps": local_representation_steps,
    }


def main(config: dict[str, Any]) -> None:
    # This function will be used to produce a config that is sent to each client to initialize their own environment
    fit_config_fn = partial(
        fit_config,
        config["batch_size"],
        config["n_server_rounds"],
        config["sample_percentage"],
        config["beta"],
        config["local_head_steps"],
        config["local_rep_steps"],
    )

    initial_model = FedRepModel(
        base_module=SequentialGlobalFeatureExtractorCifar(),
        head_module=SequentialLocalPredictionHeadCifar(),
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
        initial_parameters=get_all_model_parameters(initial_model),
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
        default="examples/fedper_example/config.yaml",
    )
    args = parser.parse_args()

    config = load_config(args.config_path)

    main(config)
