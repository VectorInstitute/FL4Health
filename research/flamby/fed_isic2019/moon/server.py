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
from research.flamby.fed_isic2019.moon.moon_model import FedIsic2019MoonModel
from research.flamby.flamby_servers.personal_server import PersonalServer
from research.flamby.utils import (
    evaluate_metrics_aggregation_fn,
    fit_config,
    fit_metrics_aggregation_fn,
    get_initial_model_parameters,
    summarize_model_info,
)


def main(config: Dict[str, Any], server_address: str, run_name: str, pretrain: bool) -> None:
    # This function will be used to produce a config that is sent to each client to initialize their own environment
    fit_config_fn = partial(
        fit_config,
        config["local_steps"],
        config["n_server_rounds"],
    )

    client_manager = SimpleClientManager()
    model = FedIsic2019MoonModel(frozen_blocks=None, turn_off_bn_tracking=False)
    summarize_model_info(model)
    if pretrain:
        dir = (
            "/ssd003/projects/aieng/public/FL_env/models/fed_isic2019/fedavg/hp_sweep_results/lr_0.001/"
            + run_name
            + "/server_best_model.pkl"
        )
        fedavg_model_state = torch.load(dir).state_dict()
        model_state = model.state_dict()
        matching_state = {}
        log(INFO, f"params: {fedavg_model_state}")
        log(INFO, f"params: {model_state}")
        for k, v in fedavg_model_state.items():
            if k in model_state:
                if v.size() == model_state[k].size():
                    matching_state[k] = v
                elif model_state[k].size()[1:] == v.size()[1:]:
                    repeat = model_state[k].size()[0] // v.size()[0]
                    original_size = tuple([1] * (len(model_state[k].size()) - 1))
                    matching_state[k] = v.repeat((repeat,) + original_size)
        log(INFO, f"matching state: {len(matching_state)}")
        model_state.update(matching_state)
        model.load_state_dict(model_state)

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
    parser.add_argument(
        "--pretrain",
        action="store_true",
        help="whether load pretrained fedavg",
    )
    args = parser.parse_args()

    config = load_config(args.config_path)
    log(INFO, f"Server Address: {args.server_address}")
    main(config, args.server_address, args.run_name, args.pretrain)
