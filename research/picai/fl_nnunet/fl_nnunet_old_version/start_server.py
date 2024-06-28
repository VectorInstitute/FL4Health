import argparse
import pickle
from functools import partial
from typing import Optional

import flwr as fl
import torch
import yaml
from batchgenerators.utilities.file_and_folder_operations import load_json
from flwr.common.parameter import ndarrays_to_parameters
from flwr.common.typing import Config
from flwr.server.client_manager import SimpleClientManager
from flwr.server.strategy import FedAvg

from examples.utils.functions import make_dict_with_epochs_or_steps
from fl4health.server.base_server import FlServer
from fl4health.utils.metric_aggregation import evaluate_metrics_aggregation_fn, fit_metrics_aggregation_fn


def get_config(
    current_server_round: int,
    nnunet_config: str,
    nnunet_plans: str,
    fold: int,
    n_server_rounds: int,
    batch_size: int,
    n_clients: int,
    local_epochs: Optional[int] = None,
    local_steps: Optional[int] = None,
) -> Config:
    nnunet_plans_dict = pickle.dumps(load_json(nnunet_plans))
    return {
        "n_clients": n_clients,
        "nnunet_config": nnunet_config,
        "nnunet_plans": nnunet_plans_dict,
        "fold": fold,
        "n_server_rounds": n_server_rounds,
        "batch_size": batch_size,
        **make_dict_with_epochs_or_steps(local_epochs, local_steps),
        "current_server_round": current_server_round,
    }


def main(config: dict) -> None:
    # Partial function with everything set except current server round
    fit_config_fn = partial(
        get_config,
        n_clients=config["n_clients"],
        nnunet_config=config["nnunet_config"],
        nnunet_plans=config["nnunet_plans"],
        fold=config["fold"],
        n_server_rounds=config["n_server_rounds"],
        batch_size=0,  # Set this to 0 because we're not using it
        local_epochs=config.get("local_epochs"),
        local_steps=config.get("local_steps"),
    )

    client_manager = SimpleClientManager()

    if config.get("starting_checkpoint"):
        model = torch.load(config["starting_checkpoint"])
        # Of course nnunet stores their pytorch models differently.
        params = ndarrays_to_parameters([val.cpu().numpy() for _, val in model["network_weights"].items()])
    else:
        raise Exception(
            "There is a bug right now where params can not be None. \
                        Therefore a starting checkpoint must be provided \
                        because I don't want to mess up my code"
        )
        # params = None

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

    server = FlServer(client_manager=client_manager, strategy=strategy)

    fl.server.start_server(
        server=server,
        server_address=config["server_address"],
        config=fl.server.ServerConfig(num_rounds=config["n_server_rounds"]),
    )

    # Shutdown server
    server.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", action="store", type=str, help="Path to the configuration file")

    args = parser.parse_args()

    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)

    main(config)
