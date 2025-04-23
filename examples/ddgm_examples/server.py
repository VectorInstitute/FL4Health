from argparse import ArgumentParser
from functools import partial

import torch
import flwr as fl
from flwr.common.typing import Config
from flwr.server import ServerConfig
from flwr.server.client_manager import SimpleClientManager

from logging import INFO
from flwr.common.logger import log

import torch.nn as nn
from examples.models.cnn_model import Net, MnistNet, FEMnistNet
from fl4health.utils.metric_aggregation import evaluate_metrics_aggregation_fn, fit_metrics_aggregation_fn
from fl4health.checkpointing.checkpointer import BestLossTorchModuleCheckpointer, LatestTorchModuleCheckpointer
from fl4health.checkpointing.server_module import BaseServerCheckpointAndStateModule
from fl4health.parameter_exchange.secure_aggregation_exchanger import SecureAggregationExchanger
from fl4health.checkpointing.checkpointer import PerRoundStateCheckpointer
from fl4health.servers.ddgm_server import DDGMServer
from examples.utils.functions import make_dict_with_epochs_or_steps


# replace later with secure aggregation strategy
from fl4health.strategies.ddgm_strategy import DDGMStrategy
from fl4health.utils.config import load_config

from examples.ddgm_examples.utils import generate_config

from fl4health.utils.parameter_extraction import get_all_model_parameters

from fl4health.privacy_mechanisms.slow_discrete_gaussian_mechanism import generate_random_sign_vector, get_exponent

from fl4health.servers.secure_aggregation_utils import vectorize_model


torch.set_default_dtype(torch.float64)
DEFAULT_MODEL_INTEGER_RANGE = 1 << 30

def fit_config(
    batch_size: int,
    current_server_round: int,
    local_epochs: int | None = None,
    local_steps: int | None = None,
) -> Config:
    return {
        **make_dict_with_epochs_or_steps(local_epochs, local_steps),
        "batch_size": batch_size,
        "current_server_round": current_server_round,
    }

if __name__ == "__main__":
    # get configurations from command line
    parser = ArgumentParser(description="DDGM server.")
    parser.add_argument(
        "--config_path",
        action="store",
        type=str,
        help="Path to configuration file. No enclosing quotes required.",
        default="examples/ddgm_examples/config.yaml",
    )
    args = parser.parse_args()
    config = load_config(args.config_path)

    # privacy settings 
    privacy_settings = {
        'enable_dp': config['enable_dp'],
        'clipping_bound': config['clipping_bound'],
        'granularity': config['granularity'],
        'model_integer_range': 1 << config['model_integer_range_exponent'],   
        'noise_multiplier': config['noise_multiplier'],
        'bias': config['bias'],
    }

    # global model (server side)
    model: nn.Module
    if config['dataset'] == 'mnist':
        model = MnistNet()
    elif config['dataset'] == 'femnist':
        model = FEMnistNet()
        log(INFO, f"Init Server Model with size {sum(p.numel() for p in model.parameters())}")
    else:
        raise NotImplementedError

    # To facilitate checkpointing
    parameter_exchanger = SecureAggregationExchanger()
    checkpointers = [
        BestLossTorchModuleCheckpointer(config["checkpoint_path"], "best_model.pkl"),
        LatestTorchModuleCheckpointer(config["checkpoint_path"], "latest_model.pkl"),
    ]

    checkpoint_and_state_module = BaseServerCheckpointAndStateModule(
        model=model, parameter_exchanger=parameter_exchanger,model_checkpointers=checkpointers,
        # state_checkpointer=state_checkpointer
    )
    
    len_parameters = len(vectorize_model(model))
    padded_model_dim = 2**get_exponent(len_parameters)
    sign_vector = generate_random_sign_vector(dim=padded_model_dim, seed=config["sign_vector_seed"])
    
    config["model_dim"] = len_parameters

    # consumed by strategy below
    fit_config_fn = partial(
        fit_config,
        config["batch_size"],
        local_epochs=config.get("local_epochs"),
        local_steps=config.get("local_steps"),
    )

    strategy = DDGMStrategy(
        min_fit_clients=config["n_clients"],
        min_evaluate_clients=config["n_clients"],
        min_available_clients=config["n_clients"],
        on_fit_config_fn=fit_config_fn,
        on_evaluate_config_fn=fit_config_fn,
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        initial_parameters=get_all_model_parameters(model),
        ddgm_config=config,
    )
    
    

    if "model_integer_range" not in config:
        config["model_integer_range"] = DEFAULT_MODEL_INTEGER_RANGE
    # configure server
    server = DDGMServer(
        client_manager=SimpleClientManager(),
        fl_config=config,
        strategy=strategy,
        checkpoint_and_state_module=checkpoint_and_state_module,
        task_name='ddgm_examples',
        privacy_settings=privacy_settings,
        sign_vector=sign_vector,
    )

    # run server
    fl.server.start_server(
        server=server,
        server_address="0.0.0.0:8081",
        config=ServerConfig(num_rounds=config["n_server_rounds"]),
    )
