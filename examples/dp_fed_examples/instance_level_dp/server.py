import argparse
import string
from collections.abc import Sequence
from functools import partial
from random import choices
from typing import Any, Dict, Optional

import flwr as fl
import torch.nn as nn
from flwr.common.parameter import parameters_to_ndarrays
from flwr.common.typing import Config
from flwr.server.client_manager import ClientManager

from examples.models.cnn_model import Net
from examples.utils.functions import make_dict_with_epochs_or_steps
from fl4health.checkpointing.opacus_checkpointer import BestLossOpacusCheckpointer, OpacusCheckpointer
from fl4health.client_managers.poisson_sampling_manager import PoissonSamplingClientManager
from fl4health.parameter_exchange.full_exchanger import FullParameterExchanger
from fl4health.reporting.base_reporter import BaseReporter
from fl4health.server.instance_level_dp_server import InstanceLevelDpServer
from fl4health.strategies.basic_fedavg import OpacusBasicFedAvg
from fl4health.utils.config import load_config
from fl4health.utils.metric_aggregation import evaluate_metrics_aggregation_fn, fit_metrics_aggregation_fn
from fl4health.utils.privacy_utilities import map_model_to_opacus_model


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


class CifarInstanceLevelDPServerWithCheckpointing(InstanceLevelDpServer):
    def __init__(
        self,
        client_manager: ClientManager,
        model: nn.Module,
        noise_multiplier: float,
        batch_size: int,
        num_server_rounds: int,
        strategy: OpacusBasicFedAvg,
        local_epochs: Optional[int] = None,
        local_steps: Optional[int] = None,
        checkpointer: Optional[OpacusCheckpointer] = None,
        reporters: Sequence[BaseReporter] | None = None,
        delta: Optional[float] = None,
    ) -> None:
        super().__init__(
            client_manager,
            noise_multiplier,
            batch_size,
            num_server_rounds,
            strategy,
            local_epochs,
            local_steps,
            checkpointer,
            reporters,
            delta,
        )
        self.parameter_exchanger = FullParameterExchanger()
        self.server_model = model

    # Setting up a hydration method so that checkpointing can happen on the server side
    def _hydrate_model_for_checkpointing(self) -> nn.Module:
        model_ndarrays = parameters_to_ndarrays(self.parameters)
        self.parameter_exchanger.pull_parameters(model_ndarrays, self.server_model)
        return self.server_model


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

    initial_model = map_model_to_opacus_model(Net())

    # ClientManager that performs Poisson type sampling
    client_manager = PoissonSamplingClientManager()

    # Server performs simple FedAveraging with Instance Level Differential Privacy
    # Must be FedAvg sampling to ensure privacy loss is computed correctly
    strategy = OpacusBasicFedAvg(
        model=initial_model,
        fraction_fit=config["client_sampling_rate"],
        # Server waits for min_available_clients before starting FL rounds
        min_available_clients=config["n_clients"],
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        on_fit_config_fn=fit_config_fn,
        # We use the same fit config function, as nothing changes for eval
        on_evaluate_config_fn=fit_config_fn,
    )

    client_name = "".join(choices(string.ascii_uppercase, k=5))
    checkpoint_dir = "examples/dp_fed_examples/instance_level_dp/"
    checkpoint_name = f"server_{client_name}_best_model.pkl"

    server = CifarInstanceLevelDPServerWithCheckpointing(
        client_manager=client_manager,
        model=initial_model,
        checkpointer=BestLossOpacusCheckpointer(checkpoint_dir=checkpoint_dir, checkpoint_name=checkpoint_name),
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
