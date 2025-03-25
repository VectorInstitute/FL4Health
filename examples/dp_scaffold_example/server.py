import argparse
from functools import partial
from typing import Any

import flwr as fl
from flwr.common.typing import Config

from examples.models.cnn_model import MnistNet
from fl4health.client_managers.poisson_sampling_manager import PoissonSamplingClientManager
from fl4health.metrics.metric_aggregation import evaluate_metrics_aggregation_fn, fit_metrics_aggregation_fn
from fl4health.servers.scaffold_server import DPScaffoldServer
from fl4health.strategies.scaffold import OpacusScaffold
from fl4health.utils.config import load_config
from fl4health.utils.privacy_utilities import map_model_to_opacus_model


def fit_config(
    local_steps: int,
    batch_size: int,
    n_server_rounds: int,
    noise_multiplier: float,
    clipping_bound: float,
    current_round: int,
) -> Config:
    return {
        "local_steps": local_steps,
        "batch_size": batch_size,
        "n_server_rounds": n_server_rounds,
        "current_server_round": current_round,
        "clipping_bound": clipping_bound,
        "noise_multiplier": noise_multiplier,
    }


def main(config: dict[str, Any]) -> None:
    # This function will be used to produce a config that is sent to each client to initialize their own environment
    fit_config_fn = partial(
        fit_config,
        config["local_steps"],
        config["batch_size"],
        config["n_server_rounds"],
        config["noise_multiplier"],
        config["clipping_bound"],
    )

    initial_model = map_model_to_opacus_model(MnistNet())

    # Initialize Scaffold strategy to handle aggregation of weights and corresponding control variates
    strategy = OpacusScaffold(
        model=initial_model,
        fraction_fit=config["client_sampling_rate"],
        min_available_clients=config["n_clients"],
        on_fit_config_fn=fit_config_fn,
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        # We use the same fit config function, as nothing changes for eval
        on_evaluate_config_fn=fit_config_fn,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
    )

    # ClientManager that performs Poisson type sampling
    client_manager = PoissonSamplingClientManager()
    server = DPScaffoldServer(
        client_manager=client_manager,
        fl_config=config,
        noise_multiplier=config["noise_multiplier"],
        batch_size=config["batch_size"],
        local_steps=config["local_steps"],
        num_server_rounds=config["n_server_rounds"],
        strategy=strategy,
        warm_start=True,
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
        default="examples/scaffold_example/config.yaml",
    )
    args = parser.parse_args()

    config = load_config(args.config_path)

    main(config)
