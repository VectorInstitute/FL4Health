import argparse
import os
from functools import partial
from logging import INFO
from typing import Any

import flwr as fl
import torch.nn as nn
from flwr.common.logger import log
from flwr.common.parameter import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.common.typing import Config, Metrics, Parameters, Scalar
from flwr.server.client_manager import ClientManager, SimpleClientManager
from flwr.server.server import EvaluateResultsAndFailures

# Specific to fedopt
from flwr.server.strategy import FedAdagrad, FedAdam, FedAvg, FedAvgM, FedYogi, Strategy

from fl4health.checkpointing.checkpointer import BestMetricTorchCheckpointer
from fl4health.parameter_exchange.full_exchanger import FullParameterExchanger
from fl4health.servers.server import FlServer
from fl4health.utils.config import load_config
from research.gemini.delirium_models.NN import NN as delirium_model
from research.gemini.mortality_models.NN import NN as mortality_model
from research.gemini.simple_metric_aggregation import metric_aggregation, normalize_metrics


class GeminiFedOptServer(FlServer):
    def __init__(
        self,
        client_manager: ClientManager,
        client_model: nn.Module,
        strategy: Strategy | None = None,
        checkpointer: BestMetricTorchCheckpointer | None = None,
    ) -> None:
        self.client_model = client_model
        # To help with model rehydration
        self.parameter_exchanger = FullParameterExchanger()
        super().__init__(client_manager, strategy, checkpointer=checkpointer)

    def _hydrate_model_for_checkpointing(self) -> None:
        model_ndarrays = parameters_to_ndarrays(self.parameters)
        self.parameter_exchanger.pull_parameters(model_ndarrays, self.client_model)

    def _maybe_checkpoint(self, checkpoint_metric: float) -> None:
        if self.checkpointer:
            self._hydrate_model_for_checkpointing()
            self.checkpointer.maybe_checkpoint(self.client_model, checkpoint_metric)

    def evaluate_round(
        self,
        server_round: int,
        timeout: float | None,
    ) -> tuple[float | None, dict[str, Scalar], EvaluateResultsAndFailures] | None:
        # loss_aggregated is the aggregated validation per step loss
        # aggregated over each client (weighted by num examples)
        eval_round_results = super().evaluate_round(server_round, timeout)
        assert eval_round_results is not None
        loss_aggregated, metrics_aggregated, (results, failures) = eval_round_results
        assert loss_aggregated is not None
        self._maybe_checkpoint(loss_aggregated)

        return loss_aggregated, metrics_aggregated, (results, failures)


def fit_metrics_aggregation_fn(all_client_metrics: list[tuple[int, Metrics]]) -> Metrics:
    # This function is run by the server to aggregate metrics returned by each clients fit function
    # NOTE: The first value of the tuple is number of examples for FedAvg
    total_examples, aggregated_metrics = metric_aggregation(all_client_metrics)
    return normalize_metrics(total_examples, aggregated_metrics)


def evaluate_metrics_aggregation_fn(all_client_metrics: list[tuple[int, Metrics]]) -> Metrics:
    # This function is run by the server to aggregate metrics returned by each clients evaluate function
    # NOTE: The first value of the tuple is number of examples for FedAvg
    total_examples, aggregated_metrics = metric_aggregation(all_client_metrics)
    return normalize_metrics(total_examples, aggregated_metrics)


def get_initial_model_parameters(client_model: nn.Module) -> Parameters:
    # Initializing the model parameters on the server side.
    # Currently uses the Pytorch default initialization for the model parameters.

    return ndarrays_to_parameters([val.cpu().numpy() for _, val in client_model.state_dict().items()])


def fit_config(
    local_epochs: int,
    batch_size: int,
    n_server_rounds: int,
    current_round: int,
) -> Config:
    return {
        "local_epochs": local_epochs,
        "batch_size": batch_size,
        "n_server_rounds": n_server_rounds,
        "current_server_round": current_round,
    }


def main(
    config: dict[str, Any],
    server_address: str,
    checkpoint_stub: str,
    run_name: str,
    algorithm: str,
    server_learning_rate: float,
) -> None:
    # This function will be used to produce a config that is sent to each client to initialize their own environment
    # mappings = get_mappings(Path(ENCOUNTERS_FILE))
    # log(INFO, "Central feature mapping done")

    fit_config_fn = partial(
        fit_config,
        config["local_epochs"],
        config["batch_size"],
        config["n_server_rounds"],
    )

    checkpoint_dir = os.path.join(checkpoint_stub, run_name)
    checkpoint_name = "server_best_model.pkl"
    checkpointer = BestMetricTorchCheckpointer(checkpoint_dir, checkpoint_name)

    client_manager = SimpleClientManager()
    if int(config["n_clients"]) == 6:
        client_model = mortality_model(input_dim=8093, output_dim=1)
    else:
        client_model = delirium_model(input_dim=35, output_dim=1)

    # Server performs the algorithm specified by user as its server-side optimization strategy
    # Options: FedAvg, FedAdam, FedAdagrad, FedAvgM, FedYogi -> client side optimization is always SGD
    strategy_call = {
        "FedAvg": FedAvg,
        "FedAdam": FedAdam,
        "FedAdagrad": FedAdagrad,
        "FedAvgM": FedAvgM,
        "FedYogi": FedYogi,
    }
    # print("Using..",strategy_call[algorithm])
    strategy = strategy_call[algorithm](
        min_fit_clients=config["n_clients"],
        min_evaluate_clients=config["n_clients"],
        # Server waits for min_available_clients before starting FL rounds
        min_available_clients=config["n_clients"],
        on_fit_config_fn=fit_config_fn,
        # We use the same fit config function, as nothing changes for eval
        on_evaluate_config_fn=fit_config_fn,
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        initial_parameters=get_initial_model_parameters(client_model),
        eta=server_learning_rate,
    )

    server = GeminiFedOptServer(client_manager, client_model, strategy, checkpointer=checkpointer)

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
        type=str,
        help="Name of the run, model checkpoints will be saved under a subfolder with this name",
        required=True,
    )
    parser.add_argument(
        "--config_path",
        action="store",
        type=str,
        help="Path to configuration file.",
        default="fedopt/config.yaml",
    )
    parser.add_argument(
        "--algorithm",
        action="store",
        type=str,
        help="Server optimizer algorithm",
        default="FedAdam",
    )
    parser.add_argument(
        "--server_learning_rate",
        action="store",
        type=float,
        help="Learning rate for server side",
        required=True,
    )
    parser.add_argument(
        "--server_address",
        action="store",
        type=str,
        help="Server Address to be used to communicate with the clients",
        default="0.0.0.0:8080",
    )
    args = parser.parse_args()

    config = load_config(args.config_path)
    log(INFO, f"Server Address: {args.server_address}")
    main(config, args.server_address, args.artifact_dir, args.run_name, args.algorithm, args.server_learning_rate)
