import argparse
from logging import DEBUG, INFO
from pathlib import Path

import flwr as fl
import torch
from flwr.common.logger import log, update_console_handler
from flwr.common.typing import Config
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from examples.models.cnn_model import MnistNet
from fl4health.clients.flexible.base import FlexibleClient
from fl4health.mixins.personalized import PersonalizedMode, make_it_personal
from fl4health.reporting import JsonReporter
from fl4health.utils.config import narrow_dict_type
from fl4health.utils.load_data import load_mnist_data
from fl4health.utils.metrics import Accuracy
from fl4health.utils.random import set_all_random_seeds
from fl4health.utils.sampler import DirichletLabelBasedSampler


class MnistClient(FlexibleClient):
    """A simple `FlexibleClient` type that we dynamically personalize via Ditto."""

    def get_data_loaders(self, config: Config) -> tuple[DataLoader, DataLoader]:
        sample_percentage = narrow_dict_type(config, "downsampling_ratio", float)
        sampler = DirichletLabelBasedSampler(list(range(10)), sample_percentage=sample_percentage, beta=1)
        batch_size = narrow_dict_type(config, "batch_size", int)
        train_loader, val_loader, _ = load_mnist_data(self.data_path, batch_size, sampler)
        return train_loader, val_loader

    def get_model(self, config: Config) -> nn.Module:
        return MnistNet().to(self.device)

    def get_optimizer(self, config: Config) -> dict[str, Optimizer]:
        # Note that the global optimizer operates on self.global_model.parameters()
        local_optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.01)
        return {"local": local_optimizer}

    def get_criterion(self, config: Config) -> _Loss:
        return torch.nn.CrossEntropyLoss()


# Dynamically created class
MnistDittoClient = make_it_personal(MnistClient, PersonalizedMode.DITTO)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FL Client Main")
    parser.add_argument("--dataset_path", action="store", type=str, help="Path to the local dataset")
    parser.add_argument(
        "--server_address",
        action="store",
        type=str,
        help="Server Address for the clients to communicate with the server through",
        default="0.0.0.0:8080",
    )
    parser.add_argument(
        "--seed",
        action="store",
        type=int,
        help="Seed for the random number generators across python, torch, and numpy",
        required=False,
    )
    parser.add_argument(
        "--debug",
        help="[OPTIONAL] Include flag to print DEBUG logs",
        action="store_const",
        dest="log_level",
        const=DEBUG,
        default=INFO,
    )
    args = parser.parse_args()

    # Set the log level
    update_console_handler(level=args.log_level)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_path = Path(args.dataset_path)
    log(INFO, f"Device to be used: {device}")
    log(INFO, f"Server Address: {args.server_address}")

    # Set the random seed for reproducibility
    set_all_random_seeds(args.seed)

    client = MnistDittoClient(data_path, [Accuracy()], device, reporters=[JsonReporter()])
    fl.client.start_client(server_address=args.server_address, client=client.to_client())

    # Shutdown the client gracefully
    client.shutdown()
