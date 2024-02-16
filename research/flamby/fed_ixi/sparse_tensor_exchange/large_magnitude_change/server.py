import argparse
from functools import partial
from logging import INFO
from typing import Any, Dict

import flwr as fl
from flamby.datasets.fed_ixi import Baseline
from flwr.common.logger import log
from flwr.common.typing import Config
from flwr.server.client_manager import SimpleClientManager

from fl4health.strategies.fedavg_sparse_coo_tensor import FedAvgSparseCooTensor
from fl4health.utils.config import load_config
from fl4health.utils.functions import get_all_model_parameters
from fl4health.utils.random import set_all_random_seeds
from research.flamby.flamby_servers.personal_server import PersonalServer
from research.flamby.utils import evaluate_metrics_aggregation_fn, fit_metrics_aggregation_fn, summarize_model_info


def fit_config(
    local_steps: int,
    n_server_rounds: int,
    current_round: int,
) -> Config:
    config: Config = {
        "local_steps": local_steps,
        "n_server_rounds": n_server_rounds,
        "current_server_round": current_round,
    }
    return config


def main(config: Dict[str, Any], server_address: str) -> None:
    # This function will be used to produce a config that is sent to each client to initialize their own environment
    fit_config_fn = partial(
        fit_config,
        config["local_steps"],
        config["n_server_rounds"],
    )

    client_manager = SimpleClientManager()

    # NOTE: We set the out_channels_first_layer to 12 rather than the default of 8. This roughly doubles the size of
    # the baseline model to be used (1106520 DOF). This is to allow for a fair parameter comparison with FENDA and APFL
    model = Baseline(out_channels_first_layer=12)
    summarize_model_info(model)

    # Server performs per-layer averaging as its server-side aggregation strategy
    strategy = FedAvgSparseCooTensor(
        # Server waits for min_available_clients before starting FL rounds
        min_available_clients=config["n_clients"],
        on_fit_config_fn=fit_config_fn,
        # We use the same fit config function, as nothing changes for eval
        on_evaluate_config_fn=fit_config_fn,
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        initial_parameters=get_all_model_parameters(model),
    )

    server = PersonalServer(client_manager, strategy)

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
        default=47,
    )
    args = parser.parse_args()

    config = load_config(args.config_path)
    log(INFO, f"Server Address: {args.server_address}")

    # Set the random seed for reproducibility
    set_all_random_seeds(args.seed)

    main(config, args.server_address)
