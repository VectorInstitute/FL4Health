import argparse
from pathlib import Path

from typing import Tuple, Callable

import flwr as fl
import torch
import torch.nn as nn
from flwr.common.typing import Config
from torch.optim import Optimizer
from torch.utils.data import DataLoader

import torchvision.transforms as transforms

from examples.models.cnn_model import (
    SslEncoder,
    SslPredictionHead,
    SslProjectionHead
)
from fl4health.clients.basic_client import BasicClient
from fl4health.utils.load_data import get_cifar10_data_and_target_tensors, split_data_and_targets
from fl4health.utils.dataset import SslTensorDataset
from fl4health.model_bases.fed_ssl_base import FedSimClrModel
from fl4health.losses.contrastive_loss import ContrastiveLoss


def get_transforms() -> Tuple[Callable, Callable]:
    input_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    color_jitter = transforms.ColorJitter(
        0.8, 0.8, 0.8, 0.2
    )
    blur = transforms.GaussianBlur((3, 3), (0.1, 2.0))

    target_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([color_jitter], p=0.8),
        transforms.RandomApply([blur], p=0.5),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return input_transform, target_transform


def get_ssl_dataloader(data_dir: Path, batch_size: int) -> Tuple[DataLoader, DataLoader]:
    data, targets = get_cifar10_data_and_target_tensors(data_dir, True)
    train_data, train_targets, val_data, val_targets = split_data_and_targets(data, targets)

    input_transform, target_transform = get_transforms()
    train_ds = SslTensorDataset(train_data, train_targets, input_transform, target_transform)
    val_ds = SslTensorDataset(val_data, val_targets, input_transform, target_transform)

    train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=3)
    val_loader = DataLoader(val_ds, batch_size, shuffle=True, num_workers=3)

    return train_loader, val_loader


class SslCifarClient(BasicClient):
    def get_data_loaders(self, config: Config) -> Tuple[DataLoader, DataLoader]:
        batch_size = self.narrow_config_type(config, "batch_size", int)
        train_loader, val_loader = get_ssl_dataloader(self.data_path, batch_size)
        return train_loader, val_loader

    def get_criterion(self, config: Config) -> nn.Module:  # type: ignore
        return ContrastiveLoss(self.device)

    def get_optimizer(self, config: Config) -> Optimizer:
        return torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

    def get_model(self, config: Config) -> nn.Module:
        ssl_model = FedSimClrModel(SslEncoder(), SslProjectionHead(), SslPredictionHead())
        return ssl_model.to(self.device)

    def transform_target(self, target: torch.Tensor) -> torch.Tensor:
        return self.model(target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FL Client Main")
    parser.add_argument("--dataset_path", action="store", type=str, help="Path to the local dataset")
    args = parser.parse_args()

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_path = Path(args.dataset_path)
    client = SslCifarClient(data_path, [], DEVICE)
    fl.client.start_client(server_address="0.0.0.0:8080", client=client.to_client())
    client.shutdown()
