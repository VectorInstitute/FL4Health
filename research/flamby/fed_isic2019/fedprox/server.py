import argparse
import os
from functools import partial
from logging import INFO
from typing import Any

import flwr as fl
from flamby.datasets.fed_isic2019 import Baseline
from flwr.common.logger import log
from flwr.server.client_manager import SimpleClientManager

from fl4health.checkpointing.checkpointer import BestLossTorchModuleCheckpointer
from fl4health.checkpointing.server_module import AdaptiveConstraintServerCheckpointAndStateModule
from fl4health.metrics.metric_aggregation import evaluate_metrics_aggregation_fn, fit_metrics_aggregation_fn
from fl4health.servers.adaptive_constraint_servers.fedprox_server import FedProxServer
from fl4health.strategies.fedavg_with_adaptive_constraint import FedAvgWithAdaptiveConstraint
from fl4health.utils.config import load_config
from fl4health.utils.parameter_extraction import get_all_model_parameters
from research.flamby.utils import fit_config


def main(config: dict[str, Any], server_address: str, mu: float, checkpoint_stub: str, run_name: str) -> None:
    # This function will be used to produce a config that is sent to each client to initialize their own environment
    fit_config_fn = partial(
        fit_config,
        config["local_steps"],
        config["n_server_rounds"],
    )

    model = Baseline()

    checkpoint_dir = os.path.join(checkpoint_stub, run_name)
    checkpoint_name = "server_best_model.pkl"
    checkpointer = BestLossTorchModuleCheckpointer(checkpoint_dir, checkpoint_name)

    checkpoint_and_state_module = AdaptiveConstraintServerCheckpointAndStateModule(
        model=model, model_checkpointers=checkpointer
    )

    client_manager = SimpleClientManager()

    # Server performs simple FedAveraging as its server-side optimization strategy
    strategy = FedAvgWithAdaptiveConstraint(
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
        initial_loss_weight=mu,
    )

    server = FedProxServer(
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
    parser.add_argument("--mu", action="store", type=float, help="Mu value for the FedProx training", default=0.1)
    args = parser.parse_args()

    config = load_config(args.config_path)
    log(INFO, f"Server Address: {args.server_address}")
    main(config, args.server_address, args.mu, args.artifact_dir, args.run_name)
