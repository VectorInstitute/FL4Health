import argparse
from pathlib import Path
from typing import Tuple

import flwr as fl
import torch
import torch.nn as nn
from flwr.common import Config
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from examples.fedpca_examples.dim_reduction.mnist_model import MnistNet
from fl4health.clients.basic_client import BasicClient
from fl4health.preprocessing.pca_preprocessor import PCAPreprocessor
from fl4health.utils.dataset import MNISTDataset
from fl4health.utils.metrics import Accuracy
from fl4health.utils.sampler import DirichletLabelBasedSampler


def get_mnist_dataset(data_dir: Path, train: bool) -> MNISTDataset:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5)),
        ]
    )
    return MNISTDataset(data_dir, train=train, transform=transform)


class MnistFedPCAClient(BasicClient):
    def get_data_loaders(self, config: Config) -> Tuple[DataLoader, DataLoader]:
        batch_size = self.narrow_config_type(config, "batch_size", int)
        pca_path = Path(self.narrow_config_type(config, "pca_path", str))
        new_dimension = self.narrow_config_type(config, "new_dimension", int)
        pca_preprocessor = PCAPreprocessor(pca_path)
        sampler = DirichletLabelBasedSampler(list(range(10)), sample_percentage=0.75, beta=1.0)

        train_loader = pca_preprocessor.reduce_dimension(
            new_dimension, batch_size, True, sampler, get_mnist_dataset, self.data_path, True
        )
        val_loader = pca_preprocessor.reduce_dimension(
            new_dimension, batch_size, False, sampler, get_mnist_dataset, self.data_path, True
        )
        return train_loader, val_loader

    def get_criterion(self, config: Config) -> _Loss:
        return torch.nn.CrossEntropyLoss()

    def get_optimizer(self, config: Config) -> Optimizer:
        return torch.optim.AdamW(self.model.parameters(), lr=0.0001)

    def get_model(self, config: Config) -> nn.Module:
        new_dimension = self.narrow_config_type(config, "new_dimension", int)
        return MnistNet(input_dim=new_dimension).to(self.device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FL Client Main")
    parser.add_argument("--dataset_path", action="store", type=str, help="Path to the local dataset")
    args = parser.parse_args()

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_path = Path(args.dataset_path)

    client = MnistFedPCAClient(data_path, [Accuracy("accuracy")], DEVICE)
    fl.client.start_numpy_client(server_address="0.0.0.0:8080", client=client)
    client.shutdown()
