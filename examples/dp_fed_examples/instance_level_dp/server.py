import argparse
from functools import partial
from typing import Any, Dict, Optional

import flwr as fl
import torch.nn as nn
from flwr.common.typing import Config

from examples.models.cnn_model import Net
from examples.utils.functions import make_dict_with_epochs_or_steps
from fl4health.client_managers.poisson_sampling_manager import PoissonSamplingClientManager
from fl4health.server.instance_level_dp_server import InstanceLevelDPServer
from fl4health.strategies.basic_fedavg import BasicFedAvg
from fl4health.utils.config import load_config
from fl4health.utils.functions import get_all_model_parameters, privacy_validate_and_fix_modules
from fl4health.utils.metric_aggregation import evaluate_metrics_aggregation_fn, fit_metrics_aggregation_fn


def construct_config(
    current_round: int,
    batch_size: int,
    noise_multiplier: float,
    clipping_bound: float,
    local_epochs: Optional[int] = None,
    local_steps: Optional[int] = None,
) -> Config:
    # NOTE: a new client is created in each round
    # NOTE: The omitted variable is server_round which allows for dynamically changing the config each round
    return {
        **make_dict_with_epochs_or_steps(local_epochs, local_steps),
        "current_server_round": current_round,
        "batch_size": batch_size,
        "noise_multiplier": noise_multiplier,
        "clipping_bound": clipping_bound,
    }


def fit_config(
    batch_size: int,
    noise_multiplier: float,
    clipping_bound: float,
    current_round: int,
    local_epochs: Optional[int] = None,
    local_steps: Optional[int] = None,
) -> Config:
    return construct_config(
        current_round,
        batch_size,
        noise_multiplier,
        clipping_bound,
        local_epochs,
        local_steps,
    )


def get_model_and_validate() -> nn.Module:
    # Create the model and use the Opacus validator to replace and layers that are not privacy compliant, such as
    # BatchNorm layers. This is also done on the client side. So we enforce a match.
    initial_model = Net()
    modified_model, _ = privacy_validate_and_fix_modules(initial_model)
    return modified_model


def main(config: Dict[str, Any]) -> None:
    # This function will be used to produce a config that is sent to each client to initialize their own environment
    fit_config_fn = partial(
        fit_config,
        config["batch_size"],
        config["noise_multiplier"],
        config["clipping_bound"],
        local_epochs=config.get("local_epochs"),
        local_steps=config.get("local_steps"),
    )

    initial_model = get_model_and_validate()

    # ClientManager that performs Poisson type sampling
    client_manager = PoissonSamplingClientManager()

    # Server performs simple FedAveraging with Instance Level Differential Privacy
    # Must be FedAvg sampling to ensure privacy loss is computed correctly
    strategy = BasicFedAvg(
        fraction_fit=config["client_sampling_rate"],
        # Server waits for min_available_clients before starting FL rounds
        min_available_clients=config["n_clients"],
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        on_fit_config_fn=fit_config_fn,
        # We use the same fit config function, as nothing changes for eval
        on_evaluate_config_fn=fit_config_fn,
        # Server side weight initialization
        initial_parameters=get_all_model_parameters(initial_model),
    )

    server = InstanceLevelDPServer(
        client_manager=client_manager,
        strategy=strategy,
        noise_multiplier=config["noise_multiplier"],
        local_epochs=config.get("local_epochs"),
        local_steps=config.get("local_steps"),
        batch_size=config["batch_size"],
        num_server_rounds=config["n_server_rounds"],
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
        default="config.yaml",
    )
    args = parser.parse_args()

    config = load_config(args.config_path)
    main(config)
