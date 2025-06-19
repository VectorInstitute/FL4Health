import argparse
from logging import INFO
from pathlib import Path

import flwr as fl
import torch
from flwr.common.logger import log
from flwr.common.typing import Config
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from examples.models.cnn_model import MnistNet
from fl4health.clients.fed_prox_client import FedProxClient
from fl4health.metrics import Accuracy
from fl4health.reporting import JsonReporter, WandBReporter, WandBStepType
from fl4health.reporting.base_reporter import BaseReporter
from fl4health.utils.config import narrow_dict_type
from fl4health.utils.load_data import load_mnist_data
from fl4health.utils.random import set_all_random_seeds
from fl4health.utils.sampler import DirichletLabelBasedSampler


class MnistFedProxClient(FedProxClient):
    def get_data_loaders(self, config: Config) -> tuple[DataLoader, DataLoader]:
        sampler = DirichletLabelBasedSampler(list(range(10)), sample_percentage=0.75, beta=1)
        batch_size = narrow_dict_type(config, "batch_size", int)
        train_loader, val_loader, _ = load_mnist_data(self.data_path, batch_size, sampler)
        return train_loader, val_loader

    def get_model(self, config: Config) -> nn.Module:
        return MnistNet().to(self.device)

    def get_optimizer(self, config: Config) -> Optimizer:
        return torch.optim.AdamW(self.model.parameters(), lr=0.01)

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
    parser.add_argument(
        "--wandb_entity",
        action="store",
        type=str,
        help="Entity to be used for W and B logging. If not provided, then no W and B logging is performed.",
        required=False,
    )
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_path = Path(args.dataset_path)
    # Set the random seed for reproducibility
    set_all_random_seeds(args.seed)
    # Get wandb_entity if provided
    wandb_entity = args.wandb_entity

    log(INFO, f"Device to be used: {device}")
    log(INFO, f"Server Address: {args.server_address}")

    json_reporter = JsonReporter()
    reporters: list[BaseReporter] = [json_reporter]

    if wandb_entity:
        log(INFO, f"Weights and Biases Entity Provided: {wandb_entity}")
        # NOTE: name/id will be set automatically and are not initialized here.
        wandb_reporter = WandBReporter(
            WandBStepType.ROUND,
            project="FL4Health",  # Name of the project under which everything should be logged
            group="FedProx Experiment",  # Group under which each of the FL run logging will be stored
            entity=wandb_entity,  # WandB user name
            tags=["Test", "FedProx"],
            job_type="client",
            notes="Testing WB reporting",
        )
        reporters.append(wandb_reporter)

    client = MnistFedProxClient(data_path, [Accuracy()], device, reporters=reporters)
    fl.client.start_client(server_address=args.server_address, client=client.to_client())

    # Shutdown the client gracefully
    client.shutdown()
