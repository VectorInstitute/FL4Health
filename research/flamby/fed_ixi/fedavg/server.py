import argparse
import os
from functools import partial
from logging import INFO
from typing import Any

import flwr as fl
from flamby.datasets.fed_ixi import Baseline
from flwr.common.logger import log
from flwr.server.client_manager import SimpleClientManager
from flwr.server.strategy import FedAvg

from fl4health.checkpointing.checkpointer import BestLossTorchModuleCheckpointer, LatestTorchModuleCheckpointer
from fl4health.checkpointing.server_module import BaseServerCheckpointAndStateModule
from fl4health.parameter_exchange.full_exchanger import FullParameterExchanger
from fl4health.utils.config import load_config
from fl4health.utils.metric_aggregation import evaluate_metrics_aggregation_fn, fit_metrics_aggregation_fn
from fl4health.utils.parameter_extraction import get_all_model_parameters
from research.flamby.flamby_servers.full_exchange_server import FullExchangeServer
from research.flamby.utils import fit_config, summarize_model_info


def main(config: dict[str, Any], server_address: str, checkpoint_stub: str, run_name: str) -> None:
    # This function will be used to produce a config that is sent to each client to initialize their own environment
    fit_config_fn = partial(
        fit_config,
        config["local_steps"],
        config["n_server_rounds"],
    )

    # NOTE: We set the out_channels_first_layer to 12 rather than the default of 8. This roughly doubles the size of
    # the baseline model to be used (1106520 DOF). This is to allow for a fair parameter comparison with FENDA and APFL
    model = Baseline(out_channels_first_layer=12)
    summarize_model_info(model)

    checkpoint_dir = os.path.join(checkpoint_stub, run_name)
    checkpoint_name = "server_best_model.pkl"
    federated_checkpointing: bool = config.get("federated_checkpointing", True)
    log(INFO, f"Performing Federated Checkpointing: {federated_checkpointing}")
    checkpointer = (
        BestLossTorchModuleCheckpointer(checkpoint_dir, checkpoint_name)
        if federated_checkpointing
        else LatestTorchModuleCheckpointer(checkpoint_dir, checkpoint_name)
    )
    checkpoint_and_state_module = BaseServerCheckpointAndStateModule(
        model=model, parameter_exchanger=FullParameterExchanger(), model_checkpointers=checkpointer
    )

    client_manager = SimpleClientManager()

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
        initial_parameters=get_all_model_parameters(model),
    )

    server = FullExchangeServer(
        client_manager=client_manager,
        fl_config=config,
        strategy=strategy,
        checkpoint_and_state_module=checkpoint_and_state_module,
    )

    fl.server.start_server(
        server=server,
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=config["n_server_rounds"]),
    )

    if federated_checkpointing:
        assert isinstance(checkpointer, BestLossTorchModuleCheckpointer)
        log(INFO, f"Best Aggregated (Weighted) Loss seen by the Server: \n{checkpointer.best_score}")

    # Shutdown the server gracefully
    server.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FL Server Main")
    parser.add_argument(
        "--artifact_dir",
        action="store",
        type=str,
        help="Path to save server artifacts such as logs and model checkpoints",
        required=True,
    )
    parser.add_argument(
        "--run_name",
        action="store",
        help="Name of the run, model checkpoints will be saved under a subfolder with this name",
        required=True,
    )
    parser.add_argument(
        "--config_path",
        action="store",
        type=str,
        help="Path to configuration file.",
        default="config.yaml",
    )
    parser.add_argument(
        "--server_address",
        action="store",
        type=str,
        help="Server Address to be used to communicate with the clients",
        default="0.0.0.0:8080",
    )
    args = parser.parse_args()

    config = load_config(args.config_path)
    log(INFO, f"Server Address: {args.server_address}")
    main(config, args.server_address, args.artifact_dir, args.run_name)
