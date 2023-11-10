import argparse
from functools import partial
from logging import INFO
from typing import Any, Dict, Optional

import flwr as fl
from flwr.common.logger import log
from flwr.common.parameter import ndarrays_to_parameters
from flwr.common.typing import Config, Parameters
from flwr.server.client_manager import SimpleClientManager

from examples.models.fenda_cnn import FendaClassifier, GlobalCnn, LocalCnn
from examples.simple_metric_aggregation import evaluate_metrics_aggregation_fn, fit_metrics_aggregation_fn
from fl4health.model_bases.fenda_base import FendaJoinMode, FendaModel
from fl4health.server.base_server import FlServer
from fl4health.strategies.basic_fedavg import BasicFedAvg
from fl4health.utils.config import load_config


def get_initial_model_parameters(initial_model: FendaModel) -> Parameters:
    # Initializing the model parameters on the server side.
    # Currently uses the Pytorch default initialization for the model parameters.
    return ndarrays_to_parameters([val.cpu().numpy() for _, val in initial_model.state_dict().items()])


def fit_config(
    local_epochs: int,
    batch_size: int,
    n_server_rounds: int,
    downsampling_ratio: float,
    current_round: int,
) -> Config:
    return {
        "local_epochs": local_epochs,
        "batch_size": batch_size,
        "n_server_rounds": n_server_rounds,
        "downsampling_ratio": downsampling_ratio,
        "current_server_round": current_round,
    }


def main(config: Dict[str, Any], server_address: str, warmed_up_dir: Optional[str], seed: Optional[int]) -> None:
    # This function will be used to produce a config that is sent to each client to initialize their own environment

    fit_config_fn = partial(
        fit_config,
        config["local_epochs"],
        config["batch_size"],
        config["n_server_rounds"],
        config["downsampling_ratio"],
    )
    model = FendaModel(
        LocalCnn(), GlobalCnn(), FendaClassifier(FendaJoinMode.CONCATENATE), warmed_up_dir=warmed_up_dir
    )
    # Server performs simple FedAveraging as its server-side optimization strategy
    strategy = BasicFedAvg(
        min_fit_clients=config["n_clients"],
        min_evaluate_clients=config["n_clients"],
        # Server waits for min_available_clients before starting FL rounds
        min_available_clients=config["n_clients"],
        on_fit_config_fn=fit_config_fn,
        # We use the same fit config function, as nothing changes for eval
        on_evaluate_config_fn=fit_config_fn,
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        initial_parameters=get_initial_model_parameters(model),
    )

    client_manager = SimpleClientManager()
    server = FlServer(client_manager, strategy, seed=seed)

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
        "--config_path",
        action="store",
        type=str,
        help="Path to configuration file.",
        default="examples/fenda_example/config.yaml",
    )
    parser.add_argument(
        "--server_address",
        action="store",
        type=str,
        help="Server Address to be used to communicate with the clients",
        default="0.0.0.0:8080",
    )
    parser.add_argument(
        "--warm_up_dir",
        action="store",
        help="Dir to save warm up checkpoint file",
        required=False,
    )
    parser.add_argument(
        "--seed",
        action="store",
        type=int,
        help="Seed for the random number generator",
        required=True,
    )
    args = parser.parse_args()

    config = load_config(args.config_path)
    log(INFO, f"Server Address: {args.server_address}")
    main(config, args.server_address, args.warm_up_dir, args.seed)
