import argparse
import os
from functools import partial
from logging import INFO
from typing import Any, Dict, List, Optional, Tuple

import flwr as fl
import numpy as np
import torch.nn as nn
from flamby.datasets.fed_isic2019 import Baseline
from flwr.common.logger import log
from flwr.common.parameter import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.common.typing import Config, Metrics, Parameters, Scalar
from flwr.server.client_manager import ClientManager
from flwr.server.server import EvaluateResultsAndFailures
from flwr.server.strategy import Strategy

from examples.simple_metric_aggregation import metric_aggregation, normalize_metrics
from fl4health.checkpointing.checkpointer import BestMetricTorchCheckpointer
from fl4health.client_managers.fixed_without_replacement_manager import FixedSamplingWithoutReplacementClientManager
from fl4health.parameter_exchange.packing_exchanger import ParameterExchangerWithPacking
from fl4health.parameter_exchange.parameter_packer import ParameterPackerWithControlVariates
from fl4health.server.base_server import FlServer
from fl4health.strategies.scaffold import Scaffold
from fl4health.utils.config import load_config


class FedIsic2019ScaffoldServer(FlServer):
    def __init__(
        self,
        client_manager: ClientManager,
        client_model: nn.Module,
        strategy: Optional[Strategy] = None,
        checkpointer: Optional[BestMetricTorchCheckpointer] = None,
    ) -> None:
        self.client_model = client_model
        # To help with model rehydration
        model_size = len(self.client_model.state_dict())
        self.parameter_exchanger = ParameterExchangerWithPacking(ParameterPackerWithControlVariates(model_size))
        super().__init__(client_manager, strategy, checkpointer=checkpointer)

    def _hydrate_model_for_checkpointing(self) -> None:
        packed_parameters = parameters_to_ndarrays(self.parameters)
        # Don't need the control variates for checkpointing.
        model_ndarrays, _ = self.parameter_exchanger.unpack_parameters(packed_parameters)
        self.parameter_exchanger.pull_parameters(model_ndarrays, self.client_model)

    def _maybe_checkpoint(self, checkpoint_metric: float) -> None:
        if self.checkpointer:
            self._hydrate_model_for_checkpointing()
            self.checkpointer.maybe_checkpoint(self.client_model, checkpoint_metric)

    def evaluate_round(
        self,
        server_round: int,
        timeout: Optional[float],
    ) -> Optional[Tuple[Optional[float], Dict[str, Scalar], EvaluateResultsAndFailures]]:
        # loss_aggregated is the aggregated validation per step loss
        # aggregated over each client (weighted by num examples)
        eval_round_results = super().evaluate_round(server_round, timeout)
        assert eval_round_results is not None
        loss_aggregated, metrics_aggregated, (results, failures) = eval_round_results
        assert loss_aggregated is not None
        self._maybe_checkpoint(loss_aggregated)

        return loss_aggregated, metrics_aggregated, (results, failures)


def fit_metrics_aggregation_fn(all_client_metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # This function is run by the server to aggregate metrics returned by each clients fit function
    # NOTE: The first value of the tuple is number of examples for FedAvg
    total_examples, aggregated_metrics = metric_aggregation(all_client_metrics)
    return normalize_metrics(total_examples, aggregated_metrics)


def evaluate_metrics_aggregation_fn(all_client_metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # This function is run by the server to aggregate metrics returned by each clients evaluate function
    # NOTE: The first value of the tuple is number of examples for FedAvg
    total_examples, aggregated_metrics = metric_aggregation(all_client_metrics)
    return normalize_metrics(total_examples, aggregated_metrics)


def get_initial_model_parameters(client_model: nn.Module) -> Parameters:
    # Initializing the model parameters on the server side.
    model_weights = [val.cpu().numpy() for _, val in client_model.state_dict().items()]
    parameters = ndarrays_to_parameters(model_weights)
    return parameters


def get_initial_model_information(client_model: nn.Module) -> Tuple[Parameters, Parameters]:
    # Initializing the model parameters on the server side.
    model_weights = [val.cpu().numpy() for _, val in client_model.state_dict().items()]
    # Initializing the control variates to zero, as suggested in the originalq scaffold paper
    control_variates = [np.zeros_like(val.data) for val in client_model.parameters() if val.requires_grad]
    return ndarrays_to_parameters(model_weights), ndarrays_to_parameters(control_variates)


def fit_config(
    local_steps: int,
    n_server_rounds: int,
    current_round: int,
) -> Config:
    return {
        "local_steps": local_steps,
        "n_server_rounds": n_server_rounds,
        "current_server_round": current_round,
    }


def main(
    config: Dict[str, Any], server_address: str, checkpoint_stub: str, run_name: str, server_learning_rate: float
) -> None:
    # This function will be used to produce a config that is sent to each client to initialize their own environment
    fit_config_fn = partial(
        fit_config,
        config["local_steps"],
        config["n_server_rounds"],
    )

    checkpoint_dir = os.path.join(checkpoint_stub, run_name)
    checkpoint_name = "server_best_model.pkl"
    checkpointer = BestMetricTorchCheckpointer(checkpoint_dir, checkpoint_name)

    client_manager = FixedSamplingWithoutReplacementClientManager()
    client_model = Baseline()

    initial_parameters, initial_control_variates = get_initial_model_information(client_model)

    strategy = Scaffold(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        # Server waits for min_available_clients before starting FL rounds
        min_available_clients=config["n_clients"],
        on_fit_config_fn=fit_config_fn,
        # We use the same fit config function, as nothing changes for eval
        on_evaluate_config_fn=fit_config_fn,
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        initial_parameters=initial_parameters,
        learning_rate=server_learning_rate,
        initial_control_variates=initial_control_variates,
    )

    server = FedIsic2019ScaffoldServer(client_manager, client_model, strategy, checkpointer)

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
    parser.add_argument(
        "--server_learning_rate",
        action="store",
        type=float,
        help="Learning rate for server side",
        required=True,
    )
    args = parser.parse_args()

    config = load_config(args.config_path)
    log(INFO, f"Server Address: {args.server_address}")
    log(INFO, f"Server Learning Rate: {args.server_learning_rate}")
    main(config, args.server_address, args.artifact_dir, args.run_name, args.server_learning_rate)
