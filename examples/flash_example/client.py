import argparse
from pathlib import Path

import flwr as fl
import torch
import torch.nn as nn
from flwr.common.typing import Config, Tuple
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from examples.models.cnn_model import Net
from fl4health.clients.flash_client import FlashClient
from fl4health.utils.load_data import load_cifar10_data
from fl4health.utils.metrics import Accuracy


class CifarFlashClient(FlashClient):
    def get_data_loaders(self, config: Config) -> Tuple[DataLoader, DataLoader]:
        batch_size = self.narrow_config_type(config, "batch_size", int)
        train_loader, val_loader, _ = load_cifar10_data(
            self.data_path,
            batch_size,
        )
        return train_loader, val_loader

    def get_model(self, config: Config) -> nn.Module:
        return Net().to(self.device)

    def get_optimizer(self, config: Config) -> Optimizer:
        return torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

    def get_criterion(self, config: Config) -> _Loss:
        return torch.nn.CrossEntropyLoss()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FL Client Main")
    parser.add_argument("--dataset_path", action="store", type=str, help="Path to the local dataset")
    args = parser.parse_args()

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_path = Path(args.dataset_path)
    client = CifarFlashClient(data_path, [Accuracy("accuracy")], DEVICE)
    fl.client.start_client(server_address="0.0.0.0:8080", client=client.to_client())
    client.shutdown()
