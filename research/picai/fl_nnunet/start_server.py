import argparse
import json
import pickle
import warnings
from functools import partial
from pathlib import Path

from fl4health.checkpointing.checkpointer import PerRoundStateCheckpointer
from fl4health.checkpointing.server_module import NnUnetServerCheckpointAndStateModule

with warnings.catch_warnings():
    # Silence deprecation warnings from sentry sdk due to flwr and wandb
    # https://github.com/adap/flower/issues/4086
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import wandb  # noqa: F401

import flwr as fl
import torch
import yaml
from flwr.common.parameter import ndarrays_to_parameters
from flwr.common.typing import Config
from flwr.server.client_manager import SimpleClientManager
from flwr.server.strategy import FedAvg

from fl4health.parameter_exchange.full_exchanger import FullParameterExchanger
from fl4health.servers.nnunet_server import NnunetServer
from fl4health.utils.config import make_dict_with_epochs_or_steps
from fl4health.utils.metric_aggregation import evaluate_metrics_aggregation_fn, fit_metrics_aggregation_fn


def get_config(
    current_server_round: int,
    nnunet_config: str,
    n_server_rounds: int,
    batch_size: int,
    n_clients: int,
    nnunet_plans: str | None = None,
    local_epochs: int | None = None,
    local_steps: int | None = None,
) -> Config:
    # Create config
    config: Config = {
        "n_clients": n_clients,
        "nnunet_config": nnunet_config,
        "n_server_rounds": n_server_rounds,
        "batch_size": batch_size,
        **make_dict_with_epochs_or_steps(local_epochs, local_steps),
        "current_server_round": current_server_round,
    }

    # Check if plans were provided
    if nnunet_plans is not None:
        plans_bytes = pickle.dumps(json.load(open(nnunet_plans, "r")))
        config["nnunet_plans"] = plans_bytes

    return config


def main(
    config: dict,
    server_address: str,
    intermediate_server_state_dir: str | None = None,
    server_name: str | None = None,
) -> None:
    # Partial function with everything set except current server round
    fit_config_fn = partial(
        get_config,
        n_clients=config["n_clients"],
        nnunet_config=config["nnunet_config"],
        n_server_rounds=config["n_server_rounds"],
        batch_size=0,  # Set this to 0 because we're not using it
        nnunet_plans=config.get("nnunet_plans"),
        local_epochs=config.get("local_epochs"),
        local_steps=config.get("local_steps"),
    )

    if config.get("starting_checkpoint"):
        model = torch.load(config["starting_checkpoint"])
        # Of course nnunet stores their pytorch models differently.
        params = ndarrays_to_parameters([val.cpu().numpy() for _, val in model["network_weights"].items()])
    else:
        # Our nnunet server subclass fixes an issue when params was None
        # https://github.com/adap/flower/issues/3770
        params = None

    strategy = FedAvg(
        min_fit_clients=config["n_clients"],
        min_evaluate_clients=config["n_clients"],
        min_available_clients=config["n_clients"],
        on_fit_config_fn=fit_config_fn,
        on_evaluate_config_fn=fit_config_fn,  # Nothing changes for eval
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        initial_parameters=params,
    )

    checkpoint_and_state_model: NnUnetServerCheckpointAndStateModule | None
    if intermediate_server_state_dir is None:
        checkpoint_and_state_model = None
    else:
        checkpoint_and_state_model = NnUnetServerCheckpointAndStateModule(
            model=None,
            parameter_exchanger=FullParameterExchanger(),
            state_checkpointer=PerRoundStateCheckpointer(Path(intermediate_server_state_dir)),
        )

    server = NnunetServer(
        client_manager=SimpleClientManager(),
        fl_config=config,
        # The fit_config_fn contains all of the necessary information for param initialization, so we reuse it here
        on_init_parameters_config_fn=fit_config_fn,
        strategy=strategy,
        checkpoint_and_state_module=checkpoint_and_state_model,
        server_name=server_name,
    )

    fl.server.start_server(
        server=server,
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=config["n_server_rounds"]),
    )

    # Shutdown server
    server.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", action="store", type=str, help="Path to the configuration file")

    parser.add_argument(
        "--server-address",
        type=str,
        required=False,
        default="0.0.0.0:8080",
        help="""[OPTIONAL] The address to use for the server. Defaults to
        0.0.0.0:8080""",
    )
    parser.add_argument(
        "--intermediate-server-state-dir",
        type=str,
        required=False,
        default="./",
        help="""[OPTIONAL] Directory to checkpoint server state. Defaults to current directory""",
    )
    parser.add_argument(
        "--server-name",
        type=str,
        required=False,
        default="server",
        help="""[OPTIONAL] Name of the server. Defaults to server""",
    )

    args = parser.parse_args()

    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)

    main(
        config,
        args.server_address,
        intermediate_server_state_dir=args.intermediate_server_state_dir,
        server_name=args.server_name,
    )
