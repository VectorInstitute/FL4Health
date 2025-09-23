import argparse
from functools import partial
from logging import INFO
from typing import Any

import flwr as fl
from fl4health.servers.server import FlServer
from flwr.common.logger import log
from flwr.common.parameter import ndarrays_to_parameters
from flwr.common.typing import Config, Metrics, Parameters, Scalar
from flwr.server.client_manager import ClientManager, SimpleClientManager
from flwr.server.server import EvaluateResultsAndFailures
from flwr.server.strategy import FedAvg, Strategy
from torch import nn

from fl4health.utils.config import load_config
from research.gemini.delirium_models.NN import NN as delirium_model
from research.gemini.mortality_models.NN import NN as mortality_model
from research.gemini.simple_metric_aggregation import metric_aggregation, normalize_metrics


class GeminiAPFLServer(FlServer):
    def __init__(
        self,
        client_manager: ClientManager,
        strategy: Strategy | None = None,
    ) -> None:
        # APFL doesn't train a "server" model. Rather, each client trains a client specific model with some globally
        # shared weights. So we don't checkpoint a global model
        super().__init__(client_manager, strategy, checkpointer=None)
        self.best_aggregated_loss: float | None = None

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

        if self.best_aggregated_loss:
            if self.best_aggregated_loss >= loss_aggregated:
                log(
                    INFO,
                    f"Best Aggregated Loss: {self.best_aggregated_loss} "
                    f"is larger than current aggregated loss: {loss_aggregated}",
                )
                self.best_aggregated_loss = loss_aggregated
            else:
                log(
                    INFO,
                    f"Best Aggregated Loss: {self.best_aggregated_loss} "
                    f"is smaller than current aggregated loss: {loss_aggregated}",
                )
        else:
            log(INFO, f"Saving Best Aggregated Loss: {loss_aggregated} as it is currently None")
            self.best_aggregated_loss = loss_aggregated

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
    current_round: int,
) -> Config:
    return {
        "local_epochs": local_epochs,
        "batch_size": batch_size,
        "current_server_round": current_round,
    }


def main(config: dict[str, Any], server_address: str, checkpoint_stub: str, run_name: str) -> None:
    # This function will be used to produce a config that is sent to each client to initialize their own environment

    # mappings = get_mappings(Path(ENCOUNTERS_FILE))

    fit_config_fn = partial(
        fit_config,
        config["local_epochs"],
        config["batch_size"],
    )

    client_manager = SimpleClientManager()

    if int(config["n_clients"]) == 6:
        client_model = mortality_model(input_dim=8093, output_dim=1)
    else:
        client_model = delirium_model(input_dim=35, output_dim=1)

    # Server performs simple FedAveraging as its server-side optimization strategy
    strategy = FedAvg(
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
    )

    server = GeminiAPFLServer(client_manager, strategy)

    fl.server.start_server(
        server=server,
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=config["n_server_rounds"]),
    )

    log(INFO, f"Best Aggregated (Weighted) Loss seen by the Server: \n{server.best_aggregated_loss}")

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
        default="apfl/config.yaml",
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
    main(config, args.server_address, args.artifact_dir, args.run_name)
