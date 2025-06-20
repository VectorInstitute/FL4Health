from argparse import ArgumentParser
from functools import partial

import torch
import flwr as fl
from flwr.common.typing import Config
from flwr.server import ServerConfig
from flwr.server.client_manager import SimpleClientManager
from fl4health.client_managers.poisson_sampling_manager import PoissonSamplingClientManager

from logging import INFO
from flwr.common.logger import log

import torch.nn as nn
from examples.models.cnn_model import Net, MnistNet, FEMnistNet
from fl4health.utils.metric_aggregation import evaluate_metrics_aggregation_fn, fit_metrics_aggregation_fn
from fl4health.checkpointing.checkpointer import BestLossTorchModuleCheckpointer, LatestTorchModuleCheckpointer
from fl4health.checkpointing.server_module import BaseServerCheckpointAndStateModule
from fl4health.parameter_exchange.secure_aggregation_exchanger import SecureAggregationExchanger
from fl4health.servers.ddgm_server import DDGMServer
from examples.utils.functions import make_dict_with_epochs_or_steps

# replace later with secure aggregation strategy
from fl4health.strategies.ddgm_strategy import DDGMStrategy
from fl4health.utils.config import load_config

from fl4health.utils.parameter_extraction import get_all_model_parameters

from fl4health.privacy_mechanisms.slow_discrete_gaussian_mechanism import generate_random_sign_vector, get_exponent

from fl4health.servers.secure_aggregation_utils import vectorize_model

from fl4health.privacy.distributed_discrete_gaussian_accountant import get_heuristic_granularity


torch.set_default_dtype(torch.float64)

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
    print("Server script: starting up.", flush=True)

    parser = ArgumentParser(description="DDGM server.")
    parser.add_argument(
        "--config_path",
        action="store",
        type=str,
        help="Path to configuration file. No enclosing quotes required.",
        default="examples/ddgm_examples/config.yaml",
    )
    parser.add_argument(
        "--server_address",
        action="store",
        type=str,
        default="0.0.0.0:8081",
    )
    args = parser.parse_args()
    config = load_config(args.config_path)

    # privacy settings 
    privacy_settings = {
        'enable_dp': config['enable_dp'],
        'clipping_bound': config['clipping_bound'],
        'sign_vector_seed': config['sign_vector_seed'],
        'bits': config['model_integer_range_exponent'],
        'model_integer_range': 1 << config['model_integer_range_exponent'],   
        'noise_multiplier': config['noise_multiplier'],
        'bias': config['bias'],
        'n_active_clients': config['n_clients'],
        'fraction_rate': config['privacy_amplification_sampling_ratio'],
        'n_clients_round': round(config['n_clients'] * config['privacy_amplification_sampling_ratio']),
        'clipping_object': config['clipping_object'],
    }



    log(INFO, f"Num of client per round: {privacy_settings['n_clients_round']}")

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
    )
    
    len_parameters = len(vectorize_model(model))
    padded_model_dim = 2**get_exponent(len_parameters)
    sign_vector = generate_random_sign_vector(dim=padded_model_dim, seed=config["sign_vector_seed"])
    
    config["model_dim"] = len_parameters

    privacy_settings['granularity'] = get_heuristic_granularity(privacy_settings["noise_multiplier"], privacy_settings['clipping_bound'], privacy_settings['bits'], privacy_settings['n_clients_round'], padded_model_dim) 
    config.setdefault('granularity', privacy_settings['granularity'])
    

    log(INFO, f"get heuristic granularity: {config['granularity']}")


    # consumed by strategy below
    fit_config_fn = partial(
        fit_config,
        config["batch_size"],
        local_epochs=config.get("local_epochs"),
        local_steps=config.get("local_steps"),
    )

    strategy = DDGMStrategy(
        fraction_fit=privacy_settings['fraction_rate'],
        fraction_evaluate=privacy_settings['fraction_rate'],
        min_fit_clients=privacy_settings['n_clients_round'],
        min_evaluate_clients=privacy_settings['n_clients_round'],
        min_available_clients=privacy_settings['n_active_clients'],
        on_fit_config_fn=fit_config_fn,
        on_evaluate_config_fn=fit_config_fn,
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        initial_parameters=get_all_model_parameters(model),
        beta=config['momentum'],
        ddgm_config=config,
    )
    
    # configure server
    client_manager = PoissonSamplingClientManager()
    server = DDGMServer(
        client_manager=client_manager,
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
        server_address=args.server_address,
        config=ServerConfig(num_rounds=config["n_server_rounds"]),
    )
