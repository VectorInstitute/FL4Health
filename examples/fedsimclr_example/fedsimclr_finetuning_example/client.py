import argparse
from pathlib import Path
from typing import Tuple

import flwr as fl
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from flwr.common.typing import Config
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from examples.models.ssl_models import CifarSslEncoder, CifarSslPredictionHead, CifarSslProjectionHead
from fl4health.clients.basic_client import BasicClient
from fl4health.model_bases.fedsimclr_base import FedSimClrModel
from fl4health.utils.config import narrow_dict_type
from fl4health.utils.dataset import TensorDataset
from fl4health.utils.load_data import get_cifar10_data_and_target_tensors, split_data_and_targets
from fl4health.utils.metrics import Accuracy


def get_finetune_dataset(data_dir: Path, batch_size: int) -> Tuple[DataLoader, DataLoader]:
    # Select test data (ie train=False) because train data was used in the pretraining stage
    data, targets = get_cifar10_data_and_target_tensors(data_dir, train=False)
    train_data, train_targets, val_data, val_targets = split_data_and_targets(data, targets)

    input_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    train_ds = TensorDataset(train_data, train_targets, input_transform)
    val_ds = TensorDataset(val_data, val_targets, input_transform)

    train_loader = DataLoader(train_ds, batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size, shuffle=False)

    return train_loader, val_loader


class CifarClient(BasicClient):
    def get_data_loaders(self, config: Config) -> Tuple[DataLoader, DataLoader]:
        batch_size = narrow_dict_type(config, "batch_size", int)
        train_loader, val_loader = get_finetune_dataset(self.data_path, batch_size)
        return train_loader, val_loader

    def get_criterion(self, config: Config) -> _Loss:
        return torch.nn.CrossEntropyLoss()

    def get_optimizer(self, config: Config) -> Optimizer:
        return torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

    def get_model(self, config: Config) -> nn.Module:
        model = FedSimClrModel(
            encoder=CifarSslEncoder(),
            projection_head=CifarSslProjectionHead(),
            prediction_head=CifarSslPredictionHead(),
            pretrain=False,
        )
        return model.to(self.device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FL Client Main")
    parser.add_argument("--dataset_path", action="store", type=str, help="Path to the local dataset")
    args = parser.parse_args()

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_path = Path(args.dataset_path)
    client = CifarClient(data_path, [Accuracy("accuracy")], DEVICE)
    fl.client.start_client(server_address="0.0.0.0:8080", client=client.to_client())
    client.shutdown()
