import argparse
from typing import Any, Dict

import flwr as fl
import yaml
from flwr.server.client_manager import SimpleClientManager
from flwr.server.server import Server

from fl4health.strategies.fedpca import FedPCA


def load_config(config_path: str) -> Dict[str, Any]:
    """Load Configuration Dictionary"""

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def main(config: Dict[str, Any]) -> None:

    n_clients = config["n_clients"]

    server = Server(client_manager=SimpleClientManager(), strategy=FedPCA(min_fit_clients=n_clients))

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
        default="examples/fedpca_example/config.yaml",
    )
    args = parser.parse_args()

    config = load_config(args.config_path)
    main(config)
