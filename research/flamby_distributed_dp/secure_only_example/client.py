from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, Tuple

import torch
from flwr.client import start_numpy_client as RunClient
from flwr.common.typing import Config
from torch.nn import CrossEntropyLoss, Module
from torch.nn.modules.loss import _Loss
from torch.optim import SGD, Optimizer
from torch.utils.data import DataLoader

from examples.models.cnn_model import Net
from fl4health.clients.secure_client import SecureClient
from fl4health.utils.load_data import load_cifar10_data
from fl4health.utils.metrics import Accuracy

import argparse
import os
from logging import INFO
from pathlib import Path
from typing import Optional, Sequence, Tuple

import flwr as fl
import torch
import torch.nn as nn
from flamby.datasets.fed_heart_disease import BATCH_SIZE, LR, NUM_CLIENTS, Baseline, BaselineLoss
from flwr.common.logger import log
from flwr.common.typing import Config
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from fl4health.checkpointing.checkpointer import BestMetricTorchCheckpointer, TorchCheckpointer
from fl4health.clients.basic_client import BasicClient
from fl4health.utils.losses import LossMeterType
from fl4health.utils.metrics import Accuracy, Metric, MetricMeterType
from research.flamby.flamby_data_utils import construct_fed_heard_disease_train_val_datasets

torch.set_default_dtype(torch.float64)


class SecureClient(SecureClient):
    # Supply @abstractmethod implementations

    """
    model, loss, optimizer
    """

    def get_model(self, config: Config) -> Module:
        return Baseline().to(self.device)

    def get_criterion(self, config: Config) -> _Loss:
        return CrossEntropyLoss()

    def get_optimizer(self, config: Config) -> Optimizer | Dict[str, Optimizer]:
        return SGD(self.model.parameters(), lr=0.001, momentum=0.9)

    """
    training & validation data
    """

    def get_data_loaders(self, config: Config) -> Tuple[DataLoader, DataLoader]:
        batch_size = self.narrow_config_type(config, "batch_size", int)

        # sample_size is currently unused
        training_loader, training_loader, sample_size = load_cifar10_data(self.data_path, batch_size)

        return training_loader, training_loader


if __name__ == "__main__":
    # link to dataset
    parser = ArgumentParser(description="Secure aggregation client.")
    parser.add_argument(
        "--dataset_path",
        action="store",
        type=str,
        default="examples/datasets",
        help="Path to the local dataset, no need to encluse in quotes.",
    )
    args = parser.parse_args()
    data_path = Path(args.dataset_path)

    # compute resource
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # instantiate Cifar client class with SecAgg as defined above
    client = SecureClient(data_path, [Accuracy("accuracy")], DEVICE)

    # NOTE server needs to be started before clients
    RunClient(server_address="0.0.0.0:8080", client=client)
    client.shutdown()
