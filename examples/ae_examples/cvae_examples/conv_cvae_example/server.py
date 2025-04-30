import argparse
from functools import partial
from typing import Any

import flwr as fl
from flwr.common.typing import Config
from flwr.server.client_manager import SimpleClientManager
from flwr.server.strategy import FedAvg

from examples.ae_examples.cvae_examples.conv_cvae_example.models import ConvConditionalDecoder, ConvConditionalEncoder
from fl4health.checkpointing.checkpointer import BestLossTorchModuleCheckpointer
from fl4health.checkpointing.server_module import BaseServerCheckpointAndStateModule
from fl4health.metrics.metric_aggregation import evaluate_metrics_aggregation_fn, fit_metrics_aggregation_fn
from fl4health.model_bases.autoencoders_base import ConditionalVae
from fl4health.parameter_exchange.full_exchanger import FullParameterExchanger
from fl4health.servers.base_server import FlServer
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
    encoder = ConvConditionalEncoder(latent_dim=int(config["latent_dim"]))
    decoder = ConvConditionalDecoder(latent_dim=int(config["latent_dim"]))
    model = ConditionalVae(encoder=encoder, decoder=decoder)
    model_checkpoint_name = "best_CVAE_model.pkl"

    # To facilitate checkpointing
    parameter_exchanger = FullParameterExchanger()
    checkpointer = BestLossTorchModuleCheckpointer(config["checkpoint_path"], model_checkpoint_name)
    checkpoint_and_state_module = BaseServerCheckpointAndStateModule(
        model=model, parameter_exchanger=parameter_exchanger, model_checkpointers=checkpointer
    )

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
        default="examples/ae_examples/cvae_examples/conv_cvae_example/config.yaml",
    )

    args = parser.parse_args()

    config = load_config(args.config_path)

    main(config)
