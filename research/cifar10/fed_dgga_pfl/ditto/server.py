import argparse
from functools import partial
from logging import INFO
from typing import Any

import flwr as fl
from flwr.common.logger import log
from flwr.common.typing import Config

from fl4health.client_managers.fixed_sampling_client_manager import FixedSamplingClientManager
from fl4health.metrics.metric_aggregation import evaluate_metrics_aggregation_fn, fit_metrics_aggregation_fn
from fl4health.strategies.feddg_ga import FairnessMetric, FairnessMetricType
from fl4health.strategies.feddg_ga_with_adaptive_constraint import FedDgGaAdaptiveConstraint
from fl4health.utils.config import load_config
from fl4health.utils.parameter_extraction import get_all_model_parameters
from fl4health.utils.random import set_all_random_seeds
from research.cifar10.model import ConvNet
from research.cifar10.personal_server import PersonalServer


def fit_config(
    batch_size: int,
    local_epochs: int,
    n_server_rounds: int,
    n_clients: int,
    current_server_round: int,
    evaluate_after_fit: bool = False,
    pack_losses_with_val_metrics: bool = False,
) -> Config:
    return {
        "batch_size": batch_size,
        "local_epochs": local_epochs,
        "n_server_rounds": n_server_rounds,
        "n_clients": n_clients,
        "current_server_round": current_server_round,
        "evaluate_after_fit": evaluate_after_fit,
        "pack_losses_with_val_metrics": pack_losses_with_val_metrics,
    }


def main(config: dict[str, Any], server_address: str, lam: float, step_size: float) -> None:
    # This function will be used to produce a config that is sent to each client to initialize their own environment
    fit_config_fn = partial(
        fit_config,
        config["batch_size"],
        config["local_epochs"],
        config["n_server_rounds"],
        config["n_clients"],
        evaluate_after_fit=config.get("evaluate_after_fit", False),
        pack_losses_with_val_metrics=config.get("pack_losses_with_val_metrics", False),
    )

    # FixedSamplingClientManager is a requirement here because the sampling cannot
    # be different between validation and evaluation for FedDG-GA to work. FixedSamplingClientManager
    # will return the same sampling until it is told to reset, which in FedDgGaStrategy
    # is done right before fit_round.
    client_manager = FixedSamplingClientManager()
    # Initializing the model on the server side
    model = ConvNet(in_channels=3, use_bn=False, dropout=0.1, hidden=512)

    # Define a fairness metric based on the loss associated with the global Ditto model as that is the one being
    # aggregated by the server.
    ditto_fairness_metric = FairnessMetric(FairnessMetricType.CUSTOM, "val - global_loss", signal=1.0)

    # Server performs simple FedAveraging as its server-side optimization strategy
    strategy = FedDgGaAdaptiveConstraint(
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
        adjustment_weight_step_size=step_size,
        fairness_metric=ditto_fairness_metric,
    )

    server = PersonalServer(client_manager=client_manager, fl_config=config, strategy=strategy)

    fl.server.start_server(
        server=server,
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=config["n_server_rounds"]),
    )

    log(INFO, "Training Complete")
    log(INFO, f"Best Aggregated (Weighted) Loss seen by the Server: \n{server.best_aggregated_loss}")

    # Shutdown the server gracefully
    server.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FL Server Main")
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
        "--lam", action="store", type=float, help="Ditto loss weight for local model training", default=0.01
    )
    parser.add_argument(
        "--step_size",
        action="store",
        type=float,
        help="Step size for Fed-DGGA Aggregation. Must be between 0.0 and 1.0. Corresponds to d in the original paper",
        required=True,
    )
    args = parser.parse_args()

    config = load_config(args.config_path)
    log(INFO, f"Server Address: {args.server_address}")
    log(INFO, f"Lambda: {args.lam}")
    log(INFO, f"Step Size: {args.step_size}")

    # Set the random seed for reproducibility
    set_all_random_seeds(args.seed, use_deterministic_torch_algos=True, disable_torch_benchmarking=True)

    main(config, args.server_address, args.lam, args.step_size)
