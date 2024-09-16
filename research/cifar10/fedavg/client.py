import argparse
import os
from logging import INFO
from pathlib import Path
from typing import Optional, Sequence, Tuple

import flwr as fl
import torch
import torch.nn as nn
from flwr.common.logger import log
from flwr.common.typing import Config
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from fl4health.checkpointing.checkpointer import BestLossTorchCheckpointer
from fl4health.checkpointing.client_module import ClientCheckpointModule
from fl4health.clients.basic_client import BasicClient
from fl4health.utils.config import narrow_config_type
from fl4health.utils.load_data import load_cifar10_data, load_cifar10_test_data
from fl4health.utils.losses import LossMeterType
from fl4health.utils.metrics import Accuracy, Metric
from fl4health.utils.random import set_all_random_seeds
from fl4health.utils.sampler import DirichletLabelBasedSampler
from research.cifar10.model import ConvNet

NUM_CLIENTS = 10


class CifarFedAvgClient(BasicClient):
    def __init__(
        self,
        data_path: Path,
        metrics: Sequence[Metric],
        device: torch.device,
        client_number: int,
        learning_rate: float,
        heterogeneity_level: float,
        loss_meter_type: LossMeterType = LossMeterType.AVERAGE,
        checkpointer: Optional[ClientCheckpointModule] = None,
    ) -> None:
        super().__init__(
            data_path=data_path,
            metrics=metrics,
            device=device,
            loss_meter_type=loss_meter_type,
            checkpointer=checkpointer,
        )
        self.client_number = client_number
        self.heterogeneity_level = heterogeneity_level
        self.learning_rate: float = learning_rate

        assert 0 <= client_number < NUM_CLIENTS
        log(INFO, f"Client Name: {self.client_name}, Client Number: {self.client_number}")

    def get_data_loaders(self, config: Config) -> Tuple[DataLoader, DataLoader]:
        batch_size = narrow_config_type(config, "batch_size", int)
        n_clients = narrow_config_type(config, "n_clients", int)
        # Set client-specific hash_key for sampler to ensure heterogneous data distribution among clients
        sampler = DirichletLabelBasedSampler(
            list(range(10)),
            sample_percentage=1.0 / n_clients,
            beta=self.heterogeneity_level,
            hash_key=self.client_number,
        )
        # Set the same hash_key for the train_loader and val_loader to ensure the same data split
        # of train and validation for all clients
        train_loader, val_loader, _ = load_cifar10_data(
            self.data_path,
            batch_size,
            validation_proportion=0.2,
            sampler=sampler,
            hash_key=100,
        )
        return train_loader, val_loader

    def get_test_data_loader(self, config: Config) -> Optional[DataLoader]:
        batch_size = narrow_config_type(config, "batch_size", int)
        n_clients = narrow_config_type(config, "n_clients", int)
        sampler = DirichletLabelBasedSampler(
            list(range(10)),
            sample_percentage=1.0 / n_clients,
            beta=self.heterogeneity_level,
            hash_key=self.client_number,
        )
        test_loader, _ = load_cifar10_test_data(self.data_path, batch_size, sampler=sampler)
        return test_loader

    def get_criterion(self, config: Config) -> _Loss:
        return torch.nn.CrossEntropyLoss()

    def get_optimizer(self, config: Config) -> Optimizer:
        return torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)

    def get_model(self, config: Config) -> nn.Module:
        return ConvNet(in_channels=3).to(self.device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FL Client Main")
    parser.add_argument(
        "--artifact_dir",
        action="store",
        type=str,
        help="Path to save client artifacts such as logs and model checkpoints",
        required=True,
    )
    parser.add_argument(
        "--dataset_dir",
        action="store",
        type=str,
        help="Path to the preprocessed Cifar 10 Dataset",
        required=True,
    )
    parser.add_argument(
        "--run_name",
        action="store",
        help="Name of the run, model checkpoints will be saved under a subfolder with this name",
        required=True,
    )
    parser.add_argument(
        "--server_address",
        action="store",
        type=str,
        help="Server Address for the clients to communicate with the server through",
        default="0.0.0.0:8080",
    )
    parser.add_argument(
        "--client_number",
        action="store",
        type=int,
        help="Number of the client for dataset loading",
        required=True,
    )
    parser.add_argument(
        "--learning_rate", action="store", type=float, help="Learning rate for local optimization", default=0.1
    )
    parser.add_argument(
        "--seed",
        action="store",
        type=int,
        help="Seed for the random number generators across python, torch, and numpy",
        required=False,
    )
    parser.add_argument(
        "--beta",
        action="store",
        type=float,
        help="Heterogeneity level for the dataset",
        required=True,
    )
    args = parser.parse_args()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(INFO, f"Device to be used: {DEVICE}")
    log(INFO, f"Server Address: {args.server_address}")
    log(INFO, f"Learning Rate: {args.learning_rate}")

    # Set the random seed for reproducibility
    set_all_random_seeds(args.seed)

    checkpoint_dir = os.path.join(args.artifact_dir, args.run_name)
    checkpoint_name = f"client_{args.client_number}_best_model.pkl"
    checkpointer = ClientCheckpointModule(post_aggregation=BestLossTorchCheckpointer(checkpoint_dir, checkpoint_name))

    data_path = Path(args.dataset_dir)
    client = CifarFedAvgClient(
        data_path=data_path,
        metrics=[Accuracy("accuracy")],
        device=DEVICE,
        client_number=args.client_number,
        learning_rate=args.learning_rate,
        heterogeneity_level=args.beta,
        checkpointer=checkpointer,
    )

    fl.client.start_client(server_address=args.server_address, client=client.to_client())
    client.shutdown()
