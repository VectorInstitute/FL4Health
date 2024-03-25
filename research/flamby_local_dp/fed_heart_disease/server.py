import argparse
from functools import partial
from typing import Any, Dict, Tuple

import flwr as fl
import numpy as np
from flwr.common.parameter import ndarrays_to_parameters
from flwr.common.typing import Config, Parameters

from examples.models.cnn_model import MnistNet
from examples.simple_metric_aggregation import evaluate_metrics_aggregation_fn, fit_metrics_aggregation_fn
from fl4health.client_managers.poisson_sampling_manager import PoissonSamplingClientManager
from fl4health.server.scaffold_server import DPScaffoldLoggingServer
from fl4health.strategies.scaffold import Scaffold
from fl4health.utils.config import load_config

from flamby.datasets.fed_heart_disease import Baseline
import os 
from fl4health.checkpointing.checkpointer import BestMetricTorchCheckpointer

from research.flamby.fed_heart_disease.large_baseline import FedHeartDiseaseLargeBaseline

def get_initial_model_information() -> Tuple[Parameters, Parameters]:
    # Initializing the model parameters on the server side.
    # Currently uses the Pytorch default initialization for the model parameters.
    initial_model = FedHeartDiseaseLargeBaseline()
    model_weights = [val.cpu().numpy() for _, val in initial_model.state_dict().items()]
    # Initializing the control variates to zero, as suggested in the original scaffold paper
    control_variates = [np.zeros_like(val.data) for val in initial_model.parameters() if val.requires_grad]
    return ndarrays_to_parameters(model_weights), ndarrays_to_parameters(control_variates)


def fit_config(
    local_steps: int,
    batch_size: int,
    n_server_rounds: int,
    noise_multiplier: float,
    clipping_bound: float,
    current_round: int,
) -> Config:
    return {
        "local_steps": local_steps,
        "batch_size": batch_size,
        "n_server_rounds": n_server_rounds,
        "current_server_round": current_round,
        "clipping_bound": clipping_bound,
        "noise_multiplier": noise_multiplier,
    }


def main(config: Dict[str, Any], checkpoint_dir) -> None:
    # This function will be used to produce a config that is sent to each client to initialize their own environment
    fit_config_fn = partial(
        fit_config,
        config["local_steps"],
        config["batch_size"],
        config["n_server_rounds"],
        config["noise_multiplier"],
        config["clipping_bound"],
    )

    initial_parameters, initial_control_variates = get_initial_model_information()

    checkpoint_name = "server_best_model.pkl"
    checkpointer = BestMetricTorchCheckpointer(checkpoint_dir, checkpoint_name)

    # Initialize Scaffold strategy to handle aggregation of weights and corresponding control variates
    strategy = Scaffold(
        fraction_fit=config["client_sampling_rate"],
        min_available_clients=config["n_clients"],
        on_fit_config_fn=fit_config_fn,
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        # We use the same fit config function, as nothing changes for eval
        on_evaluate_config_fn=fit_config_fn,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        initial_parameters=initial_parameters,
        initial_control_variates=initial_control_variates,
    )

    # ClientManager that performs Poisson type sampling
    client_manager = PoissonSamplingClientManager()
    server = DPScaffoldLoggingServer(
        client_manager=client_manager,
        noise_multiplier=config["noise_multiplier"],
        batch_size=config["batch_size"],
        local_steps=config["local_steps"],
        num_server_rounds=config["n_server_rounds"],
        strategy=strategy,
        warm_start=True,
        checkpointer=checkpointer
    )

    fl.server.start_server(
        server=server,
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=config["n_server_rounds"]),
    )
    
    server.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FL Server Main")
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
        default="examples/scaffold_example/config.yaml",
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

    checkpoint_dir = os.path.join(args.artifact_dir, args.run_name)

    main(config, checkpoint_dir)
