import argparse
from collections import OrderedDict
from collections.abc import Sequence
from functools import partial
from pathlib import Path
from typing import Any

import flwr as fl
import torch
from flwr.common import Config, NDArrays, Scalar
from flwr.server.client_manager import SimpleClientManager
from torch import nn
from torch.utils.data import DataLoader

from examples.models.cnn_model import MnistNet
from fl4health.checkpointing.checkpointer import LatestTorchModuleCheckpointer
from fl4health.metrics import Accuracy
from fl4health.metrics.base_metrics import Metric
from fl4health.metrics.metric_aggregation import evaluate_metrics_aggregation_fn, fit_metrics_aggregation_fn
from fl4health.metrics.metric_managers import MetricManager
from fl4health.parameter_exchange.full_exchanger import FullParameterExchanger
from fl4health.servers.model_merge_server import ModelMergeServer
from fl4health.strategies.model_merge_strategy import ModelMergeStrategy
from fl4health.utils.config import load_config
from fl4health.utils.load_data import load_mnist_data


def fit_config(batch_size: int, _: int) -> Config:
    return {
        "batch_size": batch_size,
    }


def server_side_evaluate_fn(
    model: nn.Module,
    loader: DataLoader,
    metrics: Sequence[Metric],
    device: torch.device,
    _: int,
    parameters: NDArrays,
    config: Config,
) -> tuple[float, dict[str, Scalar]] | None:
    model.to(device)
    model.eval()
    evaluate_metric_manager = MetricManager(metrics, "evaluate")

    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)

    with torch.no_grad():
        for input, target in loader:
            input, target = input.to(device), target.to(device)
            preds = {"predictions": model(input)}
            evaluate_metric_manager.update(preds, target)

    return 0.0, evaluate_metric_manager.compute()


def main(config: dict[str, Any], data_path: Path) -> None:
    _, val_loader, _ = load_mnist_data(data_path, config["batch_size"])
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    server_side_evaluate_fn_partial = partial(server_side_evaluate_fn, MnistNet(), val_loader, [Accuracy("")], device)

    # This function will be used to produce a config that is sent to each client to initialize their own environment
    # Current server round is ignored because it is not applicable for model merging example
    fit_config_fn = partial(
        fit_config,
        config["batch_size"],
    )

    strategy = ModelMergeStrategy(
        min_fit_clients=config["n_clients"],
        min_evaluate_clients=config["n_clients"],
        min_available_clients=config["n_clients"],
        on_fit_config_fn=fit_config_fn,
        on_evaluate_config_fn=fit_config_fn,
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        evaluate_fn=server_side_evaluate_fn_partial,
    )

    checkpointer = LatestTorchModuleCheckpointer(checkpoint_dir=config["ckpt_path"], checkpoint_name="model_merge.pt")

    server = ModelMergeServer(
        client_manager=SimpleClientManager(),
        strategy=strategy,
        checkpointer=checkpointer,
        server_model=MnistNet(),
        parameter_exchanger=FullParameterExchanger(),
    )
    fl.server.start_server(
        server=server,
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=config["n_server_rounds"]),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FL Server Main")
    parser.add_argument(
        "--config_path",
        action="store",
        type=str,
        help="Path to configuration file.",
        default="examples/model_merge_example/config.yaml",
    )

    parser.add_argument(
        "--data_path",
        action="store",
        type=str,
        help="Path to server side evaluation dataset.",
        default="examples/datasets/MNIST",
    )

    args = parser.parse_args()

    config = load_config(args.config_path)

    main(config, args.data_path)
