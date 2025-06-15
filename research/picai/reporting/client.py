import argparse
from pathlib import Path

import flwr as fl
import torch
from flwr.common.typing import Config
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from examples.models.cnn_model import Net
from fl4health.clients.basic_client import BasicClient
from fl4health.metrics import Accuracy
from fl4health.reporting import WandBReporter
from fl4health.utils.config import narrow_dict_type
from fl4health.utils.load_data import load_cifar10_data, load_cifar10_test_data


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FL Client Main")
    parser.add_argument("--dataset_path", action="store", type=str, help="Path to the local dataset")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_path = Path(args.dataset_path)
    reporter = WandBReporter(
        wandb_step_type="step",
        project="test",
        entity="haider-vector-collab",
        name="CIFAR Client",
        tags=["debug"],
        group="experiment1",
        config={"dataset": "CIFAR"},
        job_type="client",
    )
    # reporter = JsonReporter()
    client = CifarClient(data_path, [Accuracy("accuracy")], device, reporters=[reporter])
    fl.client.start_client(server_address="0.0.0.0:8080", client=client.to_client())
    client.shutdown()
