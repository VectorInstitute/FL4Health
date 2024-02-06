from argparse import ArgumentParser
from functools import partial

import torch
from flwr.server import ServerConfig
from flwr.server import start_server as RunServer
from flwr.server.client_manager import SimpleClientManager

from examples.models.cnn_model import Net

from fl4health.checkpointing.checkpointer import BestMetricTorchCheckpointer
from fl4health.client_managers.base_sampling_manager import BaseFractionSamplingManager
from fl4health.parameter_exchange.secure_aggregation_exchanger import SecureAggregationExchanger
from fl4health.server.secure_aggregation_server import SecureAggregationServer

# replace later with secure aggregation strategy
from fl4health.strategies.secure_aggregation_strategy import SecureAggregationStrategy
from fl4health.utils.config import load_config

from .utils import generate_config, get_parameters

from flamby.datasets.fed_heart_disease import Baseline
from research.flamby.flamby_servers.full_exchange_server import FullExchangeServer
from research.flamby.utils import (
    evaluate_metrics_aggregation_fn,
    fit_config,
    fit_metrics_aggregation_fn,
    get_initial_model_parameters,
    summarize_model_info,
)


torch.set_default_dtype(torch.float64)
DEFAULT_MODEL_INTEGER_RANGE = 1 << 30

if __name__ == "__main__":
    # get configurations from command line
    parser = ArgumentParser(description="Secure aggregation FedHeartDisease server.")
    parser.add_argument(
        "--config_path",
        action="store",
        type=str,
        help="Path to configuration file. No enclosing quotes required.",
        default="research/flamby_distributed_dp/secure_aggregation_example/config.yaml",
    )
    args = parser.parse_args()
    config = load_config(args.config_path)

    # global model (server side)
    model = Baseline()
    summarize_model_info(model)

    # consumed by strategy below
    config_parial = partial(generate_config, config["local_epochs"], config["batch_size"])

    fit_config_fn = partial(
    fit_config,
    config["local_steps"],
    config["n_server_rounds"],
)

    strategy = SecureAggregationStrategy(
        min_fit_clients=config["n_clients"],
        min_evaluate_clients=config["n_clients"],
        min_available_clients=config["n_clients"],
        on_fit_config_fn=config_parial,
        on_evaluate_config_fn=config_parial,
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        initial_parameters=get_initial_model_parameters(model),
    )

    if "model_integer_range" not in config:
        config["model_integer_range"] = DEFAULT_MODEL_INTEGER_RANGE
    # configure server
    server = SecureAggregationServer(
        client_manager=SimpleClientManager(),
        model=model,
        parameter_exchanger=SecureAggregationExchanger(),
        wandb_reporter=None,
        strategy=strategy,
        checkpointer=BestMetricTorchCheckpointer(config["checkpoint_path"], "best_model.pkl", maximize=False),
        shamir_reconstruction_threshold=config["shamir_reconstruction_threshold"],
        model_integer_range=config["model_integer_range"],
    )

    # run server
    RunServer(
        server=server,
        server_address="0.0.0.0:8080",
        config=ServerConfig(num_rounds=config["n_server_rounds"]),
    )
