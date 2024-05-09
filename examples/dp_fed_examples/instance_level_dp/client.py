import argparse
import string
from pathlib import Path
from random import choices
from typing import Tuple

import flwr as fl
import torch
import torch.nn as nn
from flwr.common.typing import Config
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from examples.models.cnn_model import Net
from fl4health.checkpointing.client_module import ClientCheckpointModule
from fl4health.checkpointing.opacus_checkpointer import BestLossOpacusCheckpointer
from fl4health.clients.instance_level_dp_client import InstanceLevelDpClient
from fl4health.utils.load_data import load_cifar10_data
from fl4health.utils.metrics import Accuracy


class CifarClient(InstanceLevelDpClient):
    def get_data_loaders(self, config: Config) -> Tuple[DataLoader, DataLoader]:
        batch_size = self.narrow_config_type(config, "batch_size", int)
        train_loader, val_loader, _ = load_cifar10_data(self.data_path, batch_size)
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

    client_name = "".join(choices(string.ascii_uppercase, k=5))
    checkpoint_dir = "examples/dp_fed_examples/instance_level_dp/"
    checkpoint_name = f"client_{client_name}_best_model.pkl"
    post_aggregation_checkpointer = BestLossOpacusCheckpointer(
        checkpoint_dir=checkpoint_dir, checkpoint_name=checkpoint_name
    )
    checkpointer = ClientCheckpointModule(post_aggregation=post_aggregation_checkpointer)

    # Load model and data
    data_path = Path(args.dataset_path)
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    client = CifarClient(data_path, [Accuracy("accuracy")], DEVICE, checkpointer=checkpointer)
    fl.client.start_client(server_address="0.0.0.0:8080", client=client.to_client())

    client.shutdown()
