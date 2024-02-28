import argparse
import os
from functools import partial
from logging import INFO
from typing import Any, Dict

import flwr as fl
from flamby.datasets.fed_ixi import Baseline
from flwr.common.logger import log
from flwr.server.client_manager import SimpleClientManager
from flwr.server.strategy import FedAvg

from fl4health.checkpointing.checkpointer import BestMetricTorchCheckpointer
from fl4health.utils.config import load_config
from research.flamby.flamby_servers.full_exchange_server import FullExchangeServer
from research.flamby.utils import (
    evaluate_metrics_aggregation_fn,
    fit_config,
    fit_metrics_aggregation_fn,
    get_initial_model_parameters,
    summarize_model_info,
)

from fl4health.strategies.secure_aggregation_strategy import SecureAggregationStrategy
from fl4health.parameter_exchange.secure_aggregation_exchanger import SecureAggregationExchanger
from fl4health.server.secure_aggregation_server import SecureAggregationServer



def main(config: Dict[str, Any], server_address: str, checkpoint_stub: str, run_name: str, args) -> None:
    # This function will be used to produce a config that is sent to each client to initialize their own environment
    fit_config_fn = partial(
        fit_config,
        config["local_steps"],
        config["n_server_rounds"],
    )

    checkpoint_dir = os.path.join(checkpoint_stub, run_name)
    checkpoint_name = "server_best_model.pkl"
    checkpointer = BestMetricTorchCheckpointer(checkpoint_dir, checkpoint_name)

    client_manager = SimpleClientManager()

    # NOTE: We set the out_channels_first_layer to 12 rather than the default of 8. This roughly doubles the size of
    # the baseline model to be used (1106520 DOF). This is to allow for a fair parameter comparison with FENDA and APFL
    model = Baseline(out_channels_first_layer=12)
    summarize_model_info(model)

    # Server performs simple FedAveraging as its server-side optimization strategy
    strategy = SecureAggregationStrategy(
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

    privacy_settings = {
        'clipping_threshold': config['clipping_threshold'],
        'granularity': config['granularity'],
        'model_integer_range': 1 << config['model_integer_range_exponent'],   
        'noise_scale': config['noise_scale'],
        'bias': config['bias'],
    }

    # update privacy setting for tunable hyperparameter
    key, value = args.hyperparameter_name, args.hyperparameter_value
    assert key in ['clipping_threshold', 'granularity', 'noise_scale', 'bias', 'model_integer_range']
    log(INFO, f'{type(key)}, {key}, {type(value)}, {value}')
    privacy_settings[key] = value
    log(INFO, f'{privacy_settings}')

    
    server = SecureAggregationServer(
        client_manager=client_manager,
        strategy=strategy,
        model=model,
        parameter_exchanger=SecureAggregationExchanger(),
        checkpointer=checkpointer,
        privacy_settings=privacy_settings,
    )
    fl.server.start_server(
        server=server,
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=config["n_server_rounds"]),
    )

    log(INFO, f"Best Aggregated (Weighted) Loss seen by the Server: \n{checkpointer.best_metric}")

    # Shutdown the server gracefully
    server.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FL Server Main")
    parser.add_argument(
        "--artifact_dir",
        action="store",
        type=str,
        help="Path to save server artifacts such as logs and model checkpoints",
        required=True,
    )
    parser.add_argument(
        "--run_name",
        action="store",
        help="Name of the run, model checkpoints will be saved under a subfolder with this name",
        required=True,
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
    hyperparameter_options = 'clipping_threshold, granularity, noise_scale, bias, model_integer_range_exponent'
    parser.add_argument(
        "--hyperparameter_name", action="store", type=str, help=f'Tunable hyperparameter type: {hyperparameter_options}.'
    )
    parser.add_argument(
        "--hyperparameter_value", action="store", type=float, help="Tunable hyperparameter value."
    )
    args = parser.parse_args()

    config = load_config(args.config_path)
    log(INFO, f"Server Address: {args.server_address}")
    main(config, args.server_address, args.artifact_dir, args.run_name, args)
