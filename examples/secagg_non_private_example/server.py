from argparse import ArgumentParser
from functools import partial

import torch
import flwr as fl
from flwr.common.typing import Config
from flwr.server import ServerConfig
from flwr.server.client_manager import SimpleClientManager

from examples.models.cnn_model import Net
# from examples.simple_metric_aggregation import evaluate_metrics_aggregation_fn, fit_metrics_aggregation_fn
from fl4health.utils.metric_aggregation import evaluate_metrics_aggregation_fn, fit_metrics_aggregation_fn
from fl4health.checkpointing.checkpointer import BestLossTorchModuleCheckpointer
from fl4health.client_managers.base_sampling_manager import BaseFractionSamplingManager
from fl4health.parameter_exchange.secure_aggregation_exchanger import SecureAggregationExchanger
from fl4health.servers.secure_aggregation_server import SecureAggregationServer

# replace later with secure aggregation strategy
from fl4health.strategies.secure_aggregation_strategy import SecureAggregationStrategy
from fl4health.utils.config import load_config

from examples.secure_aggregation_example.utils import generate_config, get_parameters

torch.set_default_dtype(torch.float64)
DEFAULT_MODEL_INTEGER_RANGE = 1 << 30

if __name__ == "__main__":
    # get configurations from command line
    parser = ArgumentParser(description="Secure aggregation server.")
    parser.add_argument(
        "--config_path",
        action="store",
        type=str,
        help="Path to configuration file. No enclosing quotes required.",
        default="examples/secagg_non_private_example/config.yaml",
    )
    args = parser.parse_args()
    config = load_config(args.config_path)

    # privacy settings 
    privacy_settings = {
        'clipping_threshold': config['clipping_threshold'],
        'granularity': config['granularity'],
        'model_integer_range': 1 << config['model_integer_range_exponent'],   
        'noise_scale': config['noise_scale'],
        'bias': config['bias'],
    }

    # global model (server side)
    model = Net()

    # consumed by strategy below
    config_parial = partial(generate_config, config["local_epochs"], config["batch_size"])

    strategy = SecureAggregationStrategy(
        min_fit_clients=config["n_clients"],
        min_evaluate_clients=config["n_clients"],
        min_available_clients=config["n_clients"],
        on_fit_config_fn=config_parial,
        on_evaluate_config_fn=config_parial,
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        initial_parameters=get_parameters(model),
    )

    if "model_integer_range" not in config:
        config["model_integer_range"] = DEFAULT_MODEL_INTEGER_RANGE
    # configure server
    server = SecureAggregationServer(
        client_manager=SimpleClientManager(),
        fl_config=config,
        model=model,
        parameter_exchanger=SecureAggregationExchanger(),
        wandb_reporter=None,
        strategy=strategy,
        checkpointer=BestLossTorchModuleCheckpointer(config["checkpoint_path"], "best_model.pkl"),
        shamir_reconstruction_threshold=config["shamir_reconstruction_threshold"],
        task_name='secagg_non_private_example',
        privacy_settings=privacy_settings
    )

    # run server
    fl.server.start_server(
        server=server,
        server_address="0.0.0.0:8080",
        config=ServerConfig(num_rounds=config["n_server_rounds"]),
    )
