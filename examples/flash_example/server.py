import argparse
from functools import partial
from typing import Any, Dict, Optional

import flwr as fl
from flwr.common.typing import Config
from flwr.server.client_manager import SimpleClientManager

from examples.models.cnn_model import Net
from examples.utils.functions import make_dict_with_epochs_or_steps
from fl4health.server.base_server import FlServer
from fl4health.strategies.flash import Flash
from fl4health.utils.config import load_config
from fl4health.utils.metric_aggregation import evaluate_metrics_aggregation_fn, fit_metrics_aggregation_fn
from fl4health.utils.parameter_extraction import get_all_model_parameters


def fit_config(
    batch_size: int,
    current_server_round: int,
    local_epochs: Optional[int] = None,
    local_steps: Optional[int] = None,
) -> Config:
    return {
        **make_dict_with_epochs_or_steps(local_epochs, local_steps),
        "batch_size": batch_size,
        "current_server_round": current_server_round,
    }


def main(config: Dict[str, Any]) -> None:
    fit_config_fn = partial(fit_config, config["batch_size"], local_epochs=config.get("local_epochs"))

    model = Net()

    strategy = Flash(
        min_fit_clients=config["n_clients"],
        min_evaluate_clients=config["n_clients"],
        min_available_clients=config["n_clients"],
        on_fit_config_fn=fit_config_fn,
        on_evaluate_config_fn=fit_config_fn,
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        initial_parameters=get_all_model_parameters(model),
    )

    client_manager = SimpleClientManager()
    server = FlServer(client_manager, strategy)

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
        default="examples/flash_example/config.yaml",
    )
    args = parser.parse_args()

    config = load_config(args.config_path)

    main(config)
