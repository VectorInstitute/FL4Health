import argparse
from typing import Any, Dict

import flwr as fl
from flwr.server.client_manager import SimpleClientManager

from fl4health.server.model_merge_server import ModelMergeServer
from fl4health.utils.config import load_config


def main(config: Dict[str, Any]) -> None:
    server = ModelMergeServer(client_manager=SimpleClientManager())
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
        default="examples/basic_example/config.yaml",
    )
    args = parser.parse_args()

    config = load_config(args.config_path)

    main(config)
