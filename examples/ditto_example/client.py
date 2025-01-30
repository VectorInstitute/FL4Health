import argparse
from logging import INFO
from pathlib import Path

import flwr as fl
import torch
import torch.nn as nn
from flwr.common.logger import log
from flwr.common.typing import Config
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from examples.models.cnn_model import MnistNet
from fl4health.clients.ditto_client import DittoClient
from fl4health.reporting import JsonReporter
from fl4health.utils.config import narrow_dict_type
from fl4health.utils.load_data import load_mnist_data
from fl4health.utils.metrics import Accuracy
from fl4health.utils.random import set_all_random_seeds
from fl4health.utils.sampler import DirichletLabelBasedSampler


class MnistDittoClient(DittoClient):
    def get_data_loaders(self, config: Config) -> tuple[DataLoader, DataLoader]:
        sampler = DirichletLabelBasedSampler(list(range(10)), sample_percentage=0.75, beta=1)
        batch_size = narrow_dict_type(config, "batch_size", int)
        train_loader, val_loader, _ = load_mnist_data(self.data_path, batch_size, sampler)
        return train_loader, val_loader

    def get_model(self, config: Config) -> nn.Module:
        return MnistNet().to(self.device)

    def get_optimizer(self, config: Config) -> dict[str, Optimizer]:
        # Note that the global optimizer operates on self.global_model.parameters()
        global_optimizer = torch.optim.AdamW(self.global_model.parameters(), lr=0.01)
        local_optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.01)
        return {"global": global_optimizer, "local": local_optimizer}

    def get_criterion(self, config: Config) -> _Loss:
        return torch.nn.CrossEntropyLoss()


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
    args = parser.parse_args()

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
