import argparse
from pathlib import Path
from typing import Any

import flwr as fl
import torch
from flwr.common.typing import Config
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from examples.models.cnn_model import MnistNetWithBnAndFrozen
from fl4health.clients.apfl_client import ApflClient
from fl4health.metrics import Accuracy
from fl4health.model_bases.apfl_base import ApflModule
from fl4health.reporting import JsonReporter
from fl4health.utils.config import narrow_dict_type
from fl4health.utils.load_data import load_mnist_data
from fl4health.utils.random import set_all_random_seeds
from fl4health.utils.sampler import DirichletLabelBasedSampler


class MnistApflClient(ApflClient):
    def __init__(self, *args: Any, seed: int | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.seed = seed

    def get_data_loaders(self, config: Config) -> tuple[DataLoader, DataLoader]:
        batch_size = narrow_dict_type(config, "batch_size", int)
        sampler = DirichletLabelBasedSampler(list(range(10)), sample_percentage=0.75, hash_key=self.seed)
        train_loader, val_loader, _ = load_mnist_data(self.data_path, batch_size, sampler)
        return train_loader, val_loader

    def get_model(self, config: Config) -> nn.Module:
        return ApflModule(MnistNetWithBnAndFrozen()).to(self.device)

    def get_optimizer(self, config: Config) -> dict[str, Optimizer]:
        local_optimizer = torch.optim.AdamW(self.model.local_model.parameters(), lr=0.01)
        global_optimizer = torch.optim.AdamW(self.model.global_model.parameters(), lr=0.01)
        return {"local": local_optimizer, "global": global_optimizer}

    def get_criterion(self, config: Config) -> _Loss:
        return torch.nn.CrossEntropyLoss()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FL Client Main")
    parser.add_argument("--dataset_path", action="store", type=str, help="Path to the local dataset")
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

    # Set the random seed for reproducibility
    set_all_random_seeds(args.seed)

    client = MnistApflClient(data_path, [Accuracy()], device, seed=args.seed, reporters=[JsonReporter()])
    fl.client.start_client(server_address="0.0.0.0:8080", client=client.to_client())
    client.shutdown()
