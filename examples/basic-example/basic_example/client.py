from pathlib import Path

import torch
import torch.nn as nn
from basic_example.model import Net
from flwr.client import Client, ClientApp
from flwr.client.supernode.app import run_supernode
from flwr.common import Context
from flwr.common.typing import Config
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from fl4health.clients.basic_client import BasicClient
from fl4health.utils.config import narrow_dict_type
from fl4health.utils.load_data import load_cifar10_data, load_cifar10_test_data
from fl4health.utils.metrics import Accuracy


class CifarClient(BasicClient):
    def get_data_loaders(self, config: Config) -> tuple[DataLoader, DataLoader]:
        batch_size = narrow_dict_type(config, "batch_size", int)
        train_loader, val_loader, _ = load_cifar10_data(self.data_path, batch_size)
        return train_loader, val_loader

    def get_test_data_loader(self, config: Config) -> DataLoader | None:
        batch_size = narrow_dict_type(config, "batch_size", int)
        test_loader, _ = load_cifar10_test_data(self.data_path, batch_size)
        return test_loader

    def get_criterion(self, config: Config) -> _Loss:
        return torch.nn.CrossEntropyLoss()

    def get_optimizer(self, config: Config) -> Optimizer:
        return torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

    def get_model(self, config: Config) -> nn.Module:
        return Net().to(self.device)


def client_fn(context: Context) -> Client:
    dataset_path = context.node_config["dataset_path"]
    assert isinstance(dataset_path, str), "dataset_path must be a string"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return CifarClient(Path(dataset_path), [Accuracy("accuracy")], device).to_client()


app = ClientApp(client_fn=client_fn)

if __name__ == "__main__":
    run_supernode()
