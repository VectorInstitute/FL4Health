import argparse
from collections import OrderedDict
from functools import partial
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import flwr as fl
import torch
import torch.nn as nn
from flwr.common import Config, NDArrays, Scalar
from flwr.server.client_manager import SimpleClientManager
from torch.utils.data import DataLoader

from examples.models.cnn_model import MnistNet
from fl4health.server.model_merge_server import ModelMergeServer
from fl4health.strategies.model_merge_strategy import ModelMergeStrategy
from fl4health.utils.config import load_config
from fl4health.utils.load_data import load_mnist_data
from fl4health.utils.metric_aggregation import evaluate_metrics_aggregation_fn, fit_metrics_aggregation_fn
from fl4health.utils.metrics import Accuracy, Metric, MetricManager


def move_input_data_to_device(
    data: Union[torch.Tensor, Dict[str, torch.Tensor]], device: torch.device
) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Moving data to device, where data is intended to be the input to
    model's forward method.

    Args:
        data (TorchInputType): input data to the forward method of model.
        data can be of type torch.Tensor or Dict[str, torch.Tensor], and in the
        latter case, all tensors in the dictionary are moved to self.device.

    Raises:
        TypeError: raised if data is not of type torch.Tensor or Dict[str, torch.Tensor]
    """
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {key: value.to(device) for key, value in data.items()}
    else:
        raise TypeError("data must be of type torch.Tensor or Dict[str, torch.Tensor].")


def evaluate_fn(
    model: nn.Module,
    loader: DataLoader,
    metrics: Sequence[Metric],
    device: torch.device,
    round: int,
    parameters: NDArrays,
    config: Config,
) -> Optional[Tuple[float, Dict[str, Scalar]]]:
    model.to(device)
    model.eval()
    evaluate_metric_manager = MetricManager(metrics, "evaluate")

    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)

    with torch.no_grad():
        for input, target in loader:
            input, target = move_input_data_to_device(input, device), target.to(device)
            preds = {"predictions": model(input)}
            evaluate_metric_manager.update(preds, target)

    return 0.0, evaluate_metric_manager.compute()


def main(config: Dict[str, Any], data_path: Path, batch_size: int) -> None:
    _, val_loader, _ = load_mnist_data(data_path, batch_size)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    evaluate_fn_partial = partial(evaluate_fn, MnistNet(), val_loader, [Accuracy("")], device)

    strategy = ModelMergeStrategy(
        min_fit_clients=config["n_clients"],
        min_evaluate_clients=config["n_clients"],
        # Server waits for min_available_clients before starting FL rounds
        min_available_clients=config["n_clients"],
        # We use the same fit config function, as nothing changes for eval
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        evaluate_fn=evaluate_fn_partial,
    )

    server = ModelMergeServer(client_manager=SimpleClientManager(), strategy=strategy)
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
        default="examples/basic_example/config.yaml",
    )

    parser.add_argument(
        "--data_path",
        action="store",
        type=str,
        help="Path to server side evaluation dataset.",
        default="examples/datasets/MNIST",
    )

    parser.add_argument(
        "--batch_size", action="store", type=int, help="Batch size for server size evaluation set.", default=32
    )
    args = parser.parse_args()

    config = load_config(args.config_path)

    main(config, args.data_path, args.batch_size)
