import argparse
import os
from pathlib import Path
from typing import Tuple, Optional, Dict, Callable, Sequence
from logging import INFO

import flwr as fl
import torch
import torch.nn as nn
from flwr.common.typing import Config, Scalar
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from flwr.common.logger import log

from examples.cvae_dim_example.mnist_model import MnistNet
from fl4health.clients.basic_client import BasicClient
from fl4health.utils.metrics import Accuracy, Metric
from fl4health.utils.load_data import load_mnist_data
from fl4health.utils.sampler import DirichletLabelBasedSampler
from fl4health.tasks.autoencoder_trainer import CVAETrainer


class VAEDimClient(CVAETrainer, BasicClient):
    def __init__(self, data_path: Path, metrics: Sequence[Metric], DEVICE: torch.device, condition: str):
        CVAETrainer.__init__(self, condition)
        BasicClient.__init__(self, data_path, metrics, DEVICE)

    def get_data_loaders(self, config: Config) -> Tuple[DataLoader, DataLoader]:
        batch_size = self.narrow_config_type(config, "batch_size", int)
        CVAE_model_path = Path(self.narrow_config_type(config, "CVAE_model_path", str))
        sampler = DirichletLabelBasedSampler(list(range(10)), sample_percentage=0.75, beta=100)
        sampler.set_seed(42)
        transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(torch.flatten)])

        train_loader, val_loader, _ = self.reduce_dim(
            model_path=CVAE_model_path,
            load_data=load_mnist_data,
            data_path=self.data_path,
            batch_size=batch_size,
            sampler=sampler,
            transform=transform,
        )
        return train_loader, val_loader

    def get_criterion(self, config: Config) -> _Loss:
        return torch.nn.CrossEntropyLoss()

    def get_optimizer(self, config: Config) -> Optimizer:
        return torch.optim.Adam(self.model.parameters(), lr=0.001)

    def get_model(self, config: Config) -> nn.Module:
        latent_dim = self.narrow_config_type(config, "latent_dim", int)
        return MnistNet(latent_dim).to(self.device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FL Client Main")
    parser.add_argument("--dataset_path", action="store", type=str, help="Path to the local dataset")
    parser.add_argument("--condition", action="store", type=str, help="Client ID used for CVAE")
    args = parser.parse_args()

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_path = Path(args.dataset_path)
    client = VAEDimClient(data_path, [Accuracy("accuracy")], DEVICE, args.condition)
    fl.client.start_numpy_client(server_address="0.0.0.0:8080", client=client)
    client.shutdown()
