import argparse
from functools import partial
from logging import INFO
from typing import Any

import flwr as fl
from flamby.datasets.fed_heart_disease import Baseline
from flwr.common.logger import log
from flwr.server.client_manager import SimpleClientManager
from flwr.server.strategy import FedAvg

from fl4health.metrics.metric_aggregation import evaluate_metrics_aggregation_fn, fit_metrics_aggregation_fn
from fl4health.model_bases.apfl_base import ApflModule
from fl4health.utils.config import load_config
from fl4health.utils.parameter_extraction import get_all_model_parameters
from research.flamby.flamby_servers.personal_server import PersonalServer
from research.flamby.utils import fit_config, summarize_model_info


def main(config: dict[str, Any], server_address: str) -> None:
    # This function will be used to produce a config that is sent to each client to initialize their own environment
    fit_config_fn = partial(
        fit_config,
        config["local_steps"],
        config["n_server_rounds"],
    )

    client_manager = SimpleClientManager()
    model = ApflModule(Baseline())
    summarize_model_info(model)

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
    args = parser.parse_args()

    config = load_config(args.config_path)
    log(INFO, f"Server Address: {args.server_address}")
    main(config, args.server_address)
