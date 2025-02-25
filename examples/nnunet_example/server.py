import argparse
import json
import pickle
from pathlib import Path

import flwr as fl
import torch
import yaml
from flwr.common.parameter import ndarrays_to_parameters
from flwr.server.client_manager import SimpleClientManager
from flwr.server.strategy import FedAvg

from fl4health.checkpointing.checkpointer import (
    BestMetricTorchModuleCheckpointer,
    PerRoundStateCheckpointer,
)
from fl4health.checkpointing.server_module import NnUnetServerCheckpointAndStateModule
from fl4health.parameter_exchange.full_exchanger import FullParameterExchanger
from fl4health.servers.nnunet_server import NnunetServer
from fl4health.utils.config import get_config_fn
from fl4health.utils.metric_aggregation import evaluate_metrics_aggregation_fn, fit_metrics_aggregation_fn


def main(
    config: dict,
    server_address: str,
    intermediate_server_state_dir: str | None = None,
    server_name: str | None = None,
) -> None:
    # Check if plans were provided and pickle them if so
    nnunet_plans = config.pop("nnunet_plans", None)
    if nnunet_plans is not None:
        plans_bytes = pickle.dumps(json.load(open(nnunet_plans, "r")))
        config["nnunet_plans"] = plans_bytes

    # Partial function with everything set except current server round
    fit_config_fn = get_config_fn(config, batch_size=0)  # Set batch size to 0 since we don't use it

    if config.get("starting_checkpoint"):
        model = torch.load(config["starting_checkpoint"])
        # Of course nnunet stores their pytorch models differently.
        params = ndarrays_to_parameters([val.cpu().numpy() for _, val in model["network_weights"].items()])
    else:
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

    model_checkpointer = BestMetricTorchModuleCheckpointer(
        checkpoint_dir="examples/nnunet_example/",
        checkpoint_name="checkpoint_best_ema_dice.pth",
        metric="EMA_DICE",
        maximize=True,
    )

    state_checkpointer = (
        PerRoundStateCheckpointer(Path(intermediate_server_state_dir))
        if intermediate_server_state_dir is not None
        else None
    )
    checkpoint_and_state_module = NnUnetServerCheckpointAndStateModule(
        model=None,
        parameter_exchanger=FullParameterExchanger(),
        state_checkpointer=state_checkpointer,
        model_checkpointers=[model_checkpointer],
    )

    server = NnunetServer(
        client_manager=SimpleClientManager(),
        fl_config=config,
        # The fit_config_fn contains all of the necessary information for param initialization, so we reuse it here
        on_init_parameters_config_fn=fit_config_fn,
        strategy=strategy,
        checkpoint_and_state_module=checkpoint_and_state_module,
        server_name=server_name,
        accept_failures=False,
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
    parser.add_argument(
        "--config_path",
        action="store",
        type=str,
        help="Path to the configuration file. See examples/nnunet_example/README.md for more info",
    )
    parser.add_argument(
        "--server-address",
        type=str,
        required=False,
        default="0.0.0.0:8080",
        help="[OPTIONAL] The address to use for the server. Defaults to \
        0.0.0.0:8080",
    )

    parser.add_argument(
        "--intermediate-server-state-dir",
        type=str,
        required=False,
        default=None,
        help="[OPTIONAL] Directory to store server state. Defaults to None",
    )
    parser.add_argument(
        "--server-name",
        type=str,
        required=False,
        default=None,
        help="[OPTIONAL] Name of the server used as file name when \
            checkpointing server state. Defaults to \
            None, in which case the server will generate random name \
            ",
    )
    args = parser.parse_args()

    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)

    main(
        config,
        server_address=args.server_address,
        intermediate_server_state_dir=args.intermediate_server_state_dir,
        server_name=args.server_name,
    )
