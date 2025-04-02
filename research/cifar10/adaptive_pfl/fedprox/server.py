import argparse
import os
from functools import partial
from logging import INFO
from typing import Any

import flwr as fl
from flwr.common.logger import log
from flwr.common.typing import Config
from flwr.server.client_manager import SimpleClientManager

from fl4health.checkpointing.checkpointer import BestLossTorchModuleCheckpointer, LatestTorchModuleCheckpointer
from fl4health.checkpointing.server_module import AdaptiveConstraintServerCheckpointAndStateModule
from fl4health.metrics.metric_aggregation import evaluate_metrics_aggregation_fn, fit_metrics_aggregation_fn
from fl4health.servers.adaptive_constraint_servers.fedprox_server import FedProxServer
from fl4health.strategies.fedavg_with_adaptive_constraint import FedAvgWithAdaptiveConstraint
from fl4health.utils.config import load_config
from fl4health.utils.parameter_extraction import get_all_model_parameters
from fl4health.utils.random import set_all_random_seeds
from research.cifar10.model import ConvNet


def fit_config(
    batch_size: int,
    local_epochs: int,
    n_server_rounds: int,
    n_clients: int,
    current_server_round: int,
) -> Config:
    return {
        "batch_size": batch_size,
        "local_epochs": local_epochs,
        "n_server_rounds": n_server_rounds,
        "n_clients": n_clients,
        "current_server_round": current_server_round,
    }


def main(
    config: dict[str, Any],
    server_address: str,
    checkpoint_stub: str,
    run_name: str,
    lam: float,
    adapt_loss_weight: bool,
) -> None:
    # This function will be used to produce a config that is sent to each client to initialize their own environment
    fit_config_fn = partial(
        fit_config,
        config["batch_size"],
        config["local_epochs"],
        config["n_server_rounds"],
        config["n_clients"],
    )

    # Initializing the model on the server side
    model = ConvNet(in_channels=3, use_bn=False, dropout=0.1, hidden=512)

    checkpoint_dir = os.path.join(checkpoint_stub, run_name)
    best_checkpoint_name = "server_best_model.pkl"
    last_checkpoint_name = "server_last_model.pkl"
    checkpointer = [
        BestLossTorchModuleCheckpointer(checkpoint_dir, best_checkpoint_name),
        LatestTorchModuleCheckpointer(checkpoint_dir, last_checkpoint_name),
    ]
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
        initial_loss_weight=lam,
        adapt_loss_weight=adapt_loss_weight,
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

    log(INFO, "Training Complete")
    assert isinstance(checkpointer[0], BestLossTorchModuleCheckpointer)
    log(INFO, f"Best Aggregated (Weighted) Loss seen by the Server: \n{checkpointer[0].best_score}")

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
    parser.add_argument(
        "--seed",
        action="store",
        type=int,
        help="Seed for the random number generators across python, torch, and numpy",
        required=False,
    )
    parser.add_argument(
        "--lam", action="store", type=float, help="FedProx loss weight for local model training", default=0.01
    )
    parser.add_argument(
        "--use_adaptation",
        action="store_true",
        help="Whether or not the loss weight for model drift is adapted or remains fixed.",
        default=False,
    )
    args = parser.parse_args()

    config = load_config(args.config_path)
    log(INFO, f"Server Address: {args.server_address}")
    log(INFO, f"Lambda: {args.lam}")
    if args.use_adaptation:
        log(INFO, "Adapting the loss weight for model drift via model loss")

    # Set the random seed for reproducibility
    set_all_random_seeds(args.seed, use_deterministic_torch_algos=True, disable_torch_benchmarking=True)

    main(config, args.server_address, args.artifact_dir, args.run_name, args.lam, args.use_adaptation)
