import argparse
from pathlib import Path
from typing import Any

import flwr as fl

from fl4health.client_managers.fixed_without_replacement_manager import FixedSamplingByFractionClientManager
from fl4health.metrics.metric_aggregation import uniform_evaluate_metrics_aggregation_fn
from fl4health.servers.evaluate_server import EvaluateServer
from fl4health.utils.config import load_config


def main(config: dict[str, Any], server_checkpoint_path: Path | None) -> None:
    evaluate_config = {"batch_size": config["batch_size"]}

    # ClientManager that performs Poisson type sampling
    client_manager = FixedSamplingByFractionClientManager()

    server = EvaluateServer(
        client_manager=client_manager,
        fraction_evaluate=1.0,
        model_checkpoint_path=server_checkpoint_path,
        evaluate_config=evaluate_config,
        evaluate_metrics_aggregation_fn=uniform_evaluate_metrics_aggregation_fn,
        min_available_clients=config["n_clients"],
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
        default="examples/federated_eval_example/config.yaml",
    )
    parser.add_argument(
        "--checkpoint_path",
        action="store",
        type=str,
        help="Path to server model checkpoint",
        required=False,
    )
    args = parser.parse_args()

    server_checkpoint_path = Path(args.checkpoint_path) if args.checkpoint_path else None

    config = load_config(args.config_path)

    main(config, server_checkpoint_path)
