import argparse
from functools import partial
from typing import Any

import flwr as fl
from flwr.common.typing import Config

from examples.models.cnn_model import Net
from fl4health.client_managers.poisson_sampling_manager import PoissonSamplingClientManager
from fl4health.metrics.metric_aggregation import evaluate_metrics_aggregation_fn, fit_metrics_aggregation_fn
from fl4health.servers.client_level_dp_fed_avg_server import ClientLevelDPFedAvgServer
from fl4health.strategies.client_dp_fedavgm import ClientLevelDPFedAvgM
from fl4health.utils.config import load_config, make_dict_with_epochs_or_steps
from fl4health.utils.parameter_extraction import get_all_model_parameters


def construct_config(
    current_round: int,
    batch_size: int,
    adaptive_clipping: bool,
    local_epochs: int | None = None,
    local_steps: int | None = None,
) -> Config:
    # NOTE: The omitted variable is server_round which allows for dynamically changing the config each round
    return {
        **make_dict_with_epochs_or_steps(local_epochs, local_steps),
        "batch_size": batch_size,
        "adaptive_clipping": adaptive_clipping,
        "current_server_round": current_round,
    }


def fit_config(
    batch_size: int,
    adaptive_clipping: bool,
    current_round: int,
    local_epochs: int | None = None,
    local_steps: int | None = None,
) -> Config:
    return construct_config(current_round, batch_size, adaptive_clipping, local_epochs, local_steps)


def main(config: dict[str, Any]) -> None:
    # This function will be used to produce a config that is sent to each client to initialize their own environment
    fit_config_fn = partial(
        fit_config,
        config["batch_size"],
        config["adaptive_clipping"],
        local_epochs=config.get("local_epochs"),
        local_steps=config.get("local_steps"),
    )

    initial_model = Net()

    # ClientManager that performs Poisson type sampling
    client_manager = PoissonSamplingClientManager()

    # Server performs unweighted FedAveraging with client level differential privacy
    strategy = ClientLevelDPFedAvgM(
        fraction_fit=config["client_sampling_rate"],
        # Server waits for min_available_clients before starting FL rounds
        min_available_clients=config["n_clients"],
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        on_fit_config_fn=fit_config_fn,
        # We use the same fit config function, as nothing changes for eval
        on_evaluate_config_fn=fit_config_fn,
        # Server side weight initialization
        initial_parameters=get_all_model_parameters(initial_model),
        adaptive_clipping=config["adaptive_clipping"],
        server_learning_rate=config["server_learning_rate"],
        clipping_learning_rate=config["clipping_learning_rate"],
        clipping_quantile=config["clipping_quantile"],
        initial_clipping_bound=config["clipping_bound"],
        weight_noise_multiplier=config["server_noise_multiplier"],
        clipping_noise_multiplier=config["clipping_bit_noise_multiplier"],
        beta=config["server_momentum"],
        weighted_aggregation=config["weighted_averaging"],
    )
    server = ClientLevelDPFedAvgServer(
        client_manager=client_manager,
        fl_config=config,
        strategy=strategy,
        server_noise_multiplier=config["server_noise_multiplier"],
        num_server_rounds=config["n_server_rounds"],
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
        default="config.yaml",
    )
    args = parser.parse_args()

    config = load_config(args.config_path)
    main(config)
