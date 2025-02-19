import argparse
import os
from functools import partial
from logging import INFO
from typing import Any

import flwr as fl
from flwr.common.logger import log
from flwr.common.typing import Config
from flwr.server.client_manager import SimpleClientManager
from flwr.server.strategy import FedAvg

from examples.fedllm_example.model import get_model
from examples.utils.functions import make_dict_with_epochs_or_steps
from fl4health.checkpointing.checkpointer import BestLossTorchModuleCheckpointer, LatestTorchModuleCheckpointer
from fl4health.checkpointing.server_module import BaseServerCheckpointAndStateModule
from fl4health.parameter_exchange.full_exchanger import FullParameterExchangerPeft
from fl4health.reporting import JsonReporter, WandBReporter
from fl4health.servers.base_server import FlServer
from fl4health.utils.config import load_config
from fl4health.utils.metric_aggregation import evaluate_metrics_aggregation_fn, fit_metrics_aggregation_fn
from fl4health.utils.parameter_extraction import get_all_model_parameters_peft
from fl4health.utils.random import set_all_random_seeds


def fit_config(
    batch_size: int,
    n_server_rounds: int,
    current_round: int,
    reporting_config: dict[str, str] | None = None,
    training_config: dict[str, Any] | None = None,
    model_config: dict[str, Any] | None = None,
    dataset_config: dict[str, Any] | None = None,
    local_epochs: int | None = None,
    local_steps: int | None = None,
) -> Config:
    base_config: Config = {
        **make_dict_with_epochs_or_steps(local_epochs, local_steps),
        "batch_size": batch_size,
        "n_server_rounds": n_server_rounds,
        "current_server_round": current_round,
    }
    if reporting_config is not None:
        # NOTE: that name is not included, it will be set in the clients
        base_config["project"] = reporting_config.get("project", "")
        base_config["group"] = reporting_config.get("group", "")
        base_config["entity"] = reporting_config.get("entity", "")
    if training_config is not None:
        suffix = "train"
        for key, value in training_config.items():
            if not isinstance(value, dict):
                base_config[f"{suffix}#{key}"] = value
            else:
                for k, v in value.items():
                    base_config[f"{suffix}#{key}#{k}"] = v
    if model_config is not None:
        suffix = "model"
        for key, value in model_config.items():
            if not isinstance(value, dict):
                base_config[f"{suffix}#{key}"] = value
            else:
                for k, v in value.items():
                    base_config[f"{suffix}#{key}#{k}"] = v
    if dataset_config is not None:
        suffix = "dataset"
        for key, value in dataset_config.items():
            if not isinstance(value, dict):
                base_config[f"{suffix}#{key}"] = value
            else:
                for k, v in value.items():
                    base_config[f"{suffix}#{key}#{k}"] = v
    return base_config


def main(config: dict[str, Any], server_address: str, checkpoint_stub: str, run_name: str) -> None:
    # This function will be used to produce a config that is sent to each client to initialize their own environment
    fit_config_fn = partial(
        fit_config,
        config["batch_size"],
        config["n_server_rounds"],
        reporting_config=config.get("reporting_config"),
        training_config=config.get("train"),
        model_config=config.get("model"),
        dataset_config=config.get("dataset"),
        local_epochs=config.get("local_epochs"),
        local_steps=config.get("local_steps"),
    )

    cfg_model = config.get("model")
    assert cfg_model is not None, "Config must contain a 'model' key with a dictionary value."
    init_model = get_model(cfg_model)
    parameter_exchanger = FullParameterExchangerPeft()
    checkpoint_dir = os.path.join(checkpoint_stub, run_name)
    last_checkpoint_name = "server_last_model.pkl"
    checkpointers = [
        LatestTorchModuleCheckpointer(checkpoint_dir, last_checkpoint_name),
    ]

    #     checkpoint_and_state_module = BaseServerCheckpointAndStateModule(
    #         model=init_model, parameter_exchanger=parameter_exchanger, model_checkpointers=checkpointers
    #     )

    # Server performs simple FedAveraging as its server-side optimization strategy and potentially adapts the
    # FedProx proximal weight mu
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
        initial_parameters=get_all_model_parameters_peft(init_model),
    )

    json_reporter = JsonReporter()
    client_manager = SimpleClientManager()
    if "reporting_config" in config:
        wandb_reporter = WandBReporter("round", **config["reporting_config"])
        reporters = [wandb_reporter, json_reporter]
    else:
        reporters = [json_reporter]

    server = FlServer(
        client_manager=client_manager,
        fl_config=config,
        strategy=strategy,
        reporters=reporters,
        accept_failures=False,
    )
    #     checkpoint_and_state_module=checkpoint_and_state_module,

    fl.server.start_server(
        server=server,
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=config["n_server_rounds"]),
    )
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
        default="examples/fedllm_example/config.yaml",
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
    args = parser.parse_args()

    config = load_config(args.config_path)
    log(INFO, f"Server Address: {args.server_address}")

    # Set the random seed for reproducibility
    set_all_random_seeds(args.seed)

    main(config, args.server_address, args.artifact_dir, args.run_name)
