import argparse
from functools import partial
from logging import INFO
from typing import Any, Dict

import flwr as fl
from flwr.common.logger import log
from flwr.common.typing import Config
from flwr.server.client_manager import SimpleClientManager
from flwr.server.strategy import FedAvg

from fl4health.parameter_exchange.full_exchanger import FullParameterExchanger
from fl4health.utils.config import load_config
from fl4health.utils.functions import get_all_model_parameters
from fl4health.utils.metric_aggregation import evaluate_metrics_aggregation_fn, fit_metrics_aggregation_fn
from research.picai.model_utils import get_model
from research.picai.picai_server import PicaiServer


def fit_config(
    fold_id: int,
    batch_size: int,
    local_epochs: int,
    n_server_rounds: int,
    n_clients: int,
    current_round: int,
) -> Config:
    return {
        "fold_id": fold_id,
        "batch_size": batch_size,
        "local_epochs": local_epochs,
        "n_server_rounds": n_server_rounds,
        "n_clients": n_clients,
        "current_server_round": current_round,
    }


def main(config: Dict[str, Any], server_address: str, n_clients: int) -> None:
    # This function will be used to produce a config that is sent to each client to initialize their own environment
    fit_config_fn = partial(
        fit_config,
        config["fold_id"],
        config["batch_size"],
        config["local_epochs"],
        config["n_server_rounds"],
        n_clients,  # Used to inform clients how many data partitions to create
    )

    client_manager = SimpleClientManager()
    model = get_model()

    # Server performs simple FedAveraging as its server-side optimization strategy
    strategy = FedAvg(
        min_fit_clients=n_clients,
        min_evaluate_clients=n_clients,
        # Server waits for min_available_clients before starting FL rounds
        min_available_clients=n_clients,
        on_fit_config_fn=fit_config_fn,
        # We use the same fit config function, as nothing changes for eval
        on_evaluate_config_fn=fit_config_fn,
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        initial_parameters=get_all_model_parameters(model),
    )

    server = PicaiServer(
        client_manager=client_manager,
        model=model,
        parameter_exchanger=FullParameterExchanger(),
        strategy=strategy,
        intermediate_checkpoint_dir=args.artifact_dir,
    )

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
        "--artifact_dir", action="store", type=str, help="Path to dir to store run artifacts", required=True
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
        "--n_clients",
        action="store",
        type=int,
        help="The number of clients in the FL experiments",
        required=True,
    )
    args = parser.parse_args()

    config = load_config(args.config_path)
    log(INFO, f"Server Address: {args.server_address}")
    main(config, args.server_address, args.n_clients)
