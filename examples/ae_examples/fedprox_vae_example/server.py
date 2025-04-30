import argparse
from functools import partial
from typing import Any

import flwr as fl
from flwr.common.typing import Config
from flwr.server.client_manager import SimpleClientManager

from examples.ae_examples.fedprox_vae_example.models import MnistVariationalDecoder, MnistVariationalEncoder
from fl4health.checkpointing.checkpointer import BestLossTorchModuleCheckpointer
from fl4health.checkpointing.server_module import AdaptiveConstraintServerCheckpointAndStateModule
from fl4health.metrics.metric_aggregation import evaluate_metrics_aggregation_fn, fit_metrics_aggregation_fn
from fl4health.model_bases.autoencoders_base import VariationalAe
from fl4health.servers.base_server import FlServer
from fl4health.strategies.fedavg_with_adaptive_constraint import FedAvgWithAdaptiveConstraint
from fl4health.utils.config import load_config
from fl4health.utils.parameter_extraction import get_all_model_parameters


def fit_config(
    local_epochs: int,
    batch_size: int,
    latent_dim: int,
    current_server_round: int,
) -> Config:
    return {
        "local_epochs": local_epochs,
        "batch_size": batch_size,
        "latent_dim": latent_dim,
        "current_server_round": current_server_round,
    }


def main(config: dict[str, Any]) -> None:
    # This function will be used to produce a config that is sent to each client to initialize their own environment
    fit_config_fn = partial(
        fit_config,
        config["local_epochs"],
        config["batch_size"],
        config["latent_dim"],
    )

    # Initializing the model on the server side
    encoder = MnistVariationalEncoder(input_size=784, latent_dim=int(config["latent_dim"]))
    decoder = MnistVariationalDecoder(latent_dim=int(config["latent_dim"]), output_size=784)
    model = VariationalAe(encoder=encoder, decoder=decoder)
    model_checkpoint_name = "best_VAE_model.pkl"

    # To facilitate checkpointing
    checkpointer = BestLossTorchModuleCheckpointer(config["checkpoint_path"], model_checkpoint_name)

    checkpoint_and_state_module = AdaptiveConstraintServerCheckpointAndStateModule(
        model=model, model_checkpointers=checkpointer
    )

    # Server performs simple FedAveraging as its server-side optimization strategy and potentially adapts the
    # FedProx proximal weight mu
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
        adapt_loss_weight=config["adaptive_proximal_weight"],
        initial_loss_weight=config["initial_proximal_weight"],
    )

    server = FlServer(
        client_manager=SimpleClientManager(),
        fl_config=config,
        strategy=strategy,
        checkpoint_and_state_module=checkpoint_and_state_module,
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
        default="examples/ae_examples/fedprox_vae_example/config.yaml",
    )

    args = parser.parse_args()

    config = load_config(args.config_path)

    main(config)
