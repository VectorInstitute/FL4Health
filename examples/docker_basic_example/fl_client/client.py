import argparse
from collections.abc import Sequence
from pathlib import Path

import flwr as fl
import torch
import torch.nn as nn
from flwr.common.typing import Config
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from examples.models.cnn_model import Net
from fl4health.clients.basic_client import BasicClient
from fl4health.parameter_exchange.full_exchanger import FullParameterExchanger
from fl4health.utils.config import narrow_dict_type
from fl4health.utils.load_data import load_cifar10_data
from fl4health.utils.metrics import Accuracy, Metric


class CifarClient(BasicClient):
    def __init__(self, data_path: Path, metrics: Sequence[Metric], device: torch.device) -> None:
        super().__init__(data_path, metrics, device)
        self.model = Net()
        self.parameter_exchanger = FullParameterExchanger()

    def get_data_loaders(self, config: Config) -> tuple[DataLoader, DataLoader]:
        batch_size = narrow_dict_type(config, "batch_size", int)
        train_loader, val_loader, _ = load_cifar10_data(self.data_path, batch_size)
        return train_loader, val_loader

    def get_criterion(self, config: Config) -> _Loss:
        return torch.nn.CrossEntropyLoss()

    def get_optimizer(self, config: Config) -> Optimizer:
        return torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

    def get_model(self, config: Config) -> nn.Module:
        return Net().to(self.device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FL Client Main")
    parser.add_argument("--dataset_path", action="store", type=str, help="Path to the local dataset")
    args = parser.parse_args()

    # Load model and data
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_path = Path(args.dataset_path)
    client = CifarClient(data_path, [Accuracy("accuracy")], device)
    fl.client.start_client(server_address="fl_server:8080", client=client.to_client())
