import argparse
from logging import INFO
from pathlib import Path
from typing import Dict, Tuple

import flwr as fl
import torch
import torch.nn as nn
from flwr.common.logger import log
from flwr.common.typing import Config
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from examples.models.cnn_model import MnistNetWithBnAndFrozen
from fl4health.clients.apfl_client import ApflClient
from fl4health.model_bases.apfl_base import ApflModule
from fl4health.utils.load_data import load_mnist_data
from fl4health.utils.metrics import Accuracy
from fl4health.utils.sampler import DirichletLabelBasedSampler


class MnistApflClient(ApflClient):
    def get_data_loaders(self, config: Config) -> Tuple[DataLoader, DataLoader]:
        batch_size = self.narrow_config_type(config, "batch_size", int)
        sampler = DirichletLabelBasedSampler(list(range(10)), sample_percentage=0.75)
        train_loader, val_loader, _ = load_mnist_data(self.data_path, batch_size, sampler)
        return train_loader, val_loader

    def get_model(self, config: Config) -> nn.Module:
        return ApflModule(MnistNetWithBnAndFrozen()).to(self.device)

    def get_optimizer(self, config: Config) -> Dict[str, Optimizer]:
        local_optimizer = torch.optim.AdamW(self.model.local_model.parameters(), lr=0.01)
        global_optimizer = torch.optim.AdamW(self.model.global_model.parameters(), lr=0.01)
        return {"local": local_optimizer, "global": global_optimizer}

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
        help="Seed for the random number generator",
        required=False,
        default="2023",
    )
    args = parser.parse_args()

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_path = Path(args.dataset_path)
    log(INFO, f"Device to be used: {DEVICE}")
    log(INFO, f"Server Address: {args.server_address}")

    client = MnistApflClient(data_path, [Accuracy()], DEVICE, seed=args.seed)
    fl.client.start_numpy_client(server_address=args.server_address, client=client)

    # Shutdown the client gracefully
    client.shutdown()
