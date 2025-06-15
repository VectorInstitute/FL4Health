import argparse
from collections.abc import Callable
from pathlib import Path

import flwr as fl
import torch
from flwr.common.typing import Config
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision import transforms

from examples.models.ssl_models import CifarSslEncoder, CifarSslPredictionHead, CifarSslProjectionHead
from fl4health.clients.basic_client import BasicClient
from fl4health.losses.contrastive_loss import NtXentLoss
from fl4health.model_bases.fedsimclr_base import FedSimClrModel
from fl4health.utils.config import narrow_dict_type
from fl4health.utils.dataset import SslTensorDataset
from fl4health.utils.load_data import ToNumpy, get_cifar10_data_and_target_tensors, split_data_and_targets
from fl4health.utils.typing import TorchTargetType


def get_transforms() -> tuple[Callable, Callable]:
    input_transform = transforms.Compose(
        [
            ToNumpy(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
    blur = transforms.GaussianBlur((3, 3), (0.1, 2.0))

    target_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomApply([blur], p=0.5),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    return input_transform, target_transform


def get_pretrain_dataset(data_dir: Path, batch_size: int) -> tuple[DataLoader, DataLoader]:
    data, targets = get_cifar10_data_and_target_tensors(data_dir, True)
    train_data, _, val_data, _ = split_data_and_targets(data, targets)

    input_transform, target_transform = get_transforms()

    # Since we are doing self supervised learning, targets are None
    # as the target is derived from the input itself
    train_ds = SslTensorDataset(train_data, None, input_transform, target_transform)
    val_ds = SslTensorDataset(val_data, None, input_transform, target_transform)

    train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=1)
    val_loader = DataLoader(val_ds, batch_size, shuffle=False, num_workers=1)

    return train_loader, val_loader


class SslCifarClient(BasicClient):
    def get_data_loaders(self, config: Config) -> tuple[DataLoader, DataLoader]:
        batch_size = narrow_dict_type(config, "batch_size", int)
        train_loader, val_loader = get_pretrain_dataset(self.data_path, batch_size)
        return train_loader, val_loader

    def get_criterion(self, config: Config) -> nn.Module:  # type: ignore
        return NtXentLoss(self.device)

    def get_optimizer(self, config: Config) -> Optimizer:
        return torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

    def get_model(self, config: Config) -> nn.Module:
        ssl_model = FedSimClrModel(
            CifarSslEncoder(), CifarSslProjectionHead(), CifarSslPredictionHead(), pretrain=True
        )
        return ssl_model.to(self.device)

    def transform_target(self, target: TorchTargetType) -> TorchTargetType:
        return self.model(target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FL Client Main")
    parser.add_argument("--dataset_path", action="store", type=str, help="Path to the local dataset")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_path = Path(args.dataset_path)
    client = SslCifarClient(data_path, [], device)
    fl.client.start_client(server_address="0.0.0.0:8080", client=client.to_client())
    client.shutdown()
