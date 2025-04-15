from functools import partial
from typing import Any

from basic_example.model import Net
from flwr.common import Context
from flwr.common.typing import Config
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.app import run_superlink
from flwr.server.client_manager import SimpleClientManager
from flwr.server.strategy import FedAvg

from fl4health.checkpointing.checkpointer import BestLossTorchModuleCheckpointer, LatestTorchModuleCheckpointer
from fl4health.checkpointing.server_module import BaseServerCheckpointAndStateModule
from fl4health.parameter_exchange.full_exchanger import FullParameterExchanger
from fl4health.servers.base_server import FlServer
from fl4health.utils.config import make_dict_with_epochs_or_steps
from fl4health.utils.metric_aggregation import evaluate_metrics_aggregation_fn, fit_metrics_aggregation_fn
from fl4health.utils.parameter_extraction import get_all_model_parameters


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


def main(config: dict[str, Any]) -> FlServer:
    # This function will be used to produce a config that is sent to each client to initialize their own environment
    fit_config_fn = partial(
        fit_config,
        config["batch_size"],
        local_epochs=config.get("local_epochs"),
        local_steps=config.get("local_steps"),
    )

    # Initializing the model on the server side
    model = Net()
    # To facilitate checkpointing
    parameter_exchanger = FullParameterExchanger()
    checkpointers = [
        BestLossTorchModuleCheckpointer(config["checkpoint_path"], "best_model.pkl"),
        LatestTorchModuleCheckpointer(config["checkpoint_path"], "latest_model.pkl"),
    ]
    checkpoint_and_state_module = BaseServerCheckpointAndStateModule(
        model=model, parameter_exchanger=parameter_exchanger, model_checkpointers=checkpointers
    )

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
        initial_parameters=get_all_model_parameters(model),
    )

    server = FlServer(
        client_manager=SimpleClientManager(),
        fl_config=config,
        strategy=strategy,
        checkpoint_and_state_module=checkpoint_and_state_module,
        accept_failures=False,
    )

    return server


def server_fn(context: Context) -> ServerAppComponents:
    config = {
        "n_server_rounds": context.run_config["n_server_rounds"],
        "n_clients": context.run_config["n_clients"],
        "local_epochs": context.run_config["local_epochs"],
        "batch_size": context.run_config["batch_size"],
        "checkpoint_path": context.run_config["checkpoint_path"],
    }

    assert isinstance(config["n_server_rounds"], int), "n_server_rounds must be an integer"

    server = main(config)
    server_config = ServerConfig(num_rounds=config["n_server_rounds"])
    return ServerAppComponents(server=server, config=server_config)


app = ServerApp(server_fn=server_fn)

if __name__ == "__main__":
    run_superlink()
