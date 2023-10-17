import argparse
from functools import partial
from logging import INFO
from typing import Any, Dict

import flwr as fl
import torch
from flwr.common.logger import log
from flwr.server.client_manager import SimpleClientManager
from flwr.server.strategy import FedAvg

from fl4health.utils.config import load_config
from research.flamby.fed_heart_disease.fenda.fenda_model import FedHeartDiseaseFendaModel
from research.flamby.flamby_servers.personal_server import PersonalServer
from research.flamby.utils import (
    evaluate_metrics_aggregation_fn,
    fit_config_with_warmup,
    fit_metrics_aggregation_fn,
    get_initial_model_parameters,
    summarize_model_info,
)


def main(config: Dict[str, Any], server_address: str, run_name: str) -> None:
    # This function will be used to produce a config that is sent to each client to initialize their own environment
    fit_config_fn = partial(
        fit_config_with_warmup,
        config["local_steps"],
        config["n_server_rounds"],
        config["warmup_rounds"],
    )

    client_manager = SimpleClientManager()
    model = FedHeartDiseaseFendaModel()
    summarize_model_info(model)
    if config["load_fedavg_model"]:
        dir = (
            "/Users/sanaayromlou/Desktop/FL4Health/results/fed_heart_disease/fedavg/"
            + run_name
            + "/server_best_model.pkl"
        )
        fedavg_model_state = torch.load(dir).state_dict()
        model_state = model.global_module.state_dict()
        for k, v in fedavg_model_state.items():
            if k in model_state:
                if v.size() == model_state[k].size():
                    fedavg_model_state[k] = v
                elif model_state[k].size()[1:] == v.size()[1:]:
                    repeat = model_state[k].size()[0] // v.size()[0]
                    original_size = tuple([1] * (len(model_state[k].size()) - 1))
                    fedavg_model_state[k] = v.repeat((repeat,) + original_size)
                else:
                    del fedavg_model_state[k]
            else:
                del fedavg_model_state[k]
        model_state.update(fedavg_model_state)
        model.global_module.load_state_dict(model_state)

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
        initial_parameters=get_initial_model_parameters(model),
    )

    server = PersonalServer(client_manager, strategy)

    fl.server.start_server(
        server=server,
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=config["n_server_rounds"]),
    )

    log(INFO, "Training Complete")
    log(INFO, f"Best Aggregated (Weighted) Loss seen by the Server: \n{server.best_aggregated_loss}")

    # Shutdown the server gracefully
    server.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FL Server Main")
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
        "--run_name",
        action="store",
        help="Name of the run, model checkpoints will be saved under a subfolder with this name",
        required=True,
    )
    args = parser.parse_args()

    config = load_config(args.config_path)
    log(INFO, f"Server Address: {args.server_address}")
    main(config, args.server_address, args.run_name)
