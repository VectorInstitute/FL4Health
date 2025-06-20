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
from fl4health.servers.base_server import FlServer
from fl4health.strategies.basic_fedavg import BasicFedAvg
from examples.utils.functions import make_dict_with_epochs_or_steps
from fl4health.parameter_exchange.full_exchanger import FullParameterExchanger
# replace later with secure aggregation strategy
from fl4health.strategies.ddgm_strategy import DDGMStrategy
from fl4health.utils.config import load_config

from fl4health.utils.parameter_extraction import get_all_model_parameters


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

    parser = ArgumentParser(description="FL Server Main with FEMNIST data.")
    parser.add_argument(
        "--config_path",
        action="store",
        type=str,
        help="Path to configuration file. No enclosing quotes required.",
        default="examples/basic_femnist_example/config.yaml",
    )
    parser.add_argument(
        "--server_address",
        action="store",
        type=str,
        default="0.0.0.0:8081",
    )
    args = parser.parse_args()
    config = load_config(args.config_path)

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
    parameter_exchanger = FullParameterExchanger()
    checkpointers = [
        BestLossTorchModuleCheckpointer(config["checkpoint_path"], "best_model.pkl"),
        LatestTorchModuleCheckpointer(config["checkpoint_path"], "latest_model.pkl"),
    ]

    checkpoint_and_state_module = BaseServerCheckpointAndStateModule(
        model=model, parameter_exchanger=parameter_exchanger,model_checkpointers=checkpointers,
    )

    # consumed by strategy below
    fit_config_fn = partial(
        fit_config,
        config["batch_size"],
        local_epochs=config.get("local_epochs"),
        local_steps=config.get("local_steps"),
    )

    strategy = BasicFedAvg(
        fraction_fit=config['fraction_rate'],
        fraction_evaluate=config['fraction_rate'],
        min_fit_clients=config['n_clients'],
        min_evaluate_clients=config['n_clients'],
        min_available_clients=config['n_clients'],
        on_fit_config_fn=fit_config_fn,
        on_evaluate_config_fn=fit_config_fn,
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        initial_parameters=get_all_model_parameters(model),
    )
    
    # configure server
    client_manager = PoissonSamplingClientManager()
    server = FlServer(
        client_manager=client_manager,
        fl_config=config,
        strategy=strategy,
        checkpoint_and_state_module=checkpoint_and_state_module,
        accept_failures=False,
    )

    # run server
    fl.server.start_server(
        server=server,
        server_address=args.server_address,
        config=ServerConfig(num_rounds=config["n_server_rounds"]),
    )
