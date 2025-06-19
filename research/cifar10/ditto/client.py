import argparse
import os
from collections.abc import Sequence
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

from fl4health.checkpointing.checkpointer import BestLossTorchModuleCheckpointer, LatestTorchModuleCheckpointer
from fl4health.checkpointing.client_module import ClientCheckpointAndStateModule
from fl4health.clients.ditto_client import DittoClient
from fl4health.metrics import Accuracy
from fl4health.metrics.base_metrics import Metric
from fl4health.reporting.base_reporter import BaseReporter
from fl4health.utils.config import narrow_dict_type
from fl4health.utils.load_data import load_cifar10_data, load_cifar10_test_data
from fl4health.utils.losses import LossMeterType
from fl4health.utils.random import set_all_random_seeds
from fl4health.utils.sampler import DirichletLabelBasedSampler
from research.cifar10.model import ConvNet
from research.cifar10.preprocess import get_preprocessed_data, get_test_preprocessed_data


class CifarDittoClient(DittoClient):
    def __init__(
        self,
        data_path: Path,
        metrics: Sequence[Metric],
        device: torch.device,
        client_number: int,
        learning_rate: float,
        heterogeneity_level: float,
        loss_meter_type: LossMeterType = LossMeterType.AVERAGE,
        checkpoint_and_state_module: ClientCheckpointAndStateModule | None = None,
        reporters: Sequence[BaseReporter] | None = None,
        progress_bar: bool = False,
        client_name: str | None = None,
        use_partitioned_data: bool = False,
    ) -> None:
        super().__init__(
            data_path=data_path,
            metrics=metrics,
            device=device,
            loss_meter_type=loss_meter_type,
            checkpoint_and_state_module=checkpoint_and_state_module,
            reporters=reporters,
            progress_bar=progress_bar,
            client_name=client_name,
        )
        self.use_partitioned_data = use_partitioned_data
        self.client_number = client_number
        self.heterogeneity_level = heterogeneity_level
        self.learning_rate: float = learning_rate

    def setup_client(self, config: Config) -> None:
        # Check if the client number is within the range of the total number of clients
        num_clients = narrow_dict_type(config, "n_clients", int)
        assert 0 <= self.client_number < num_clients
        super().setup_client(config)

    def get_data_loaders(self, config: Config) -> tuple[DataLoader, DataLoader]:
        batch_size = narrow_dict_type(config, "batch_size", int)
        if self.use_partitioned_data:
            # The partitioned data should be generated prior to running the clients via preprocess_data function
            # in the research/cifar10/preprocess.py file
            train_loader, val_loader, _ = get_preprocessed_data(
                self.data_path, self.client_number, batch_size, self.heterogeneity_level
            )
        else:
            n_clients = narrow_dict_type(config, "n_clients", int)
            # Set client-specific hash_key for sampler to ensure heterogeneous data distribution among clients
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

    def get_test_data_loader(self, config: Config) -> DataLoader | None:
        batch_size = narrow_dict_type(config, "batch_size", int)
        if self.use_partitioned_data:
            # The partitioned data should be generated prior to running the clients via preprocess_data function
            # in the research/cifar10/preprocess.py file
            test_loader, _ = get_test_preprocessed_data(
                self.data_path, self.client_number, batch_size, self.heterogeneity_level
            )
        else:
            n_clients = narrow_dict_type(config, "n_clients", int)
            # Set client-specific hash_key for sampler to ensure heterogeneous data distribution among clients
            # Also as hash_key is same between train and test sampler, the test data distribution will be same
            # as the train data distribution
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

    def get_optimizer(self, config: Config) -> dict[str, Optimizer]:
        # Following the implementation in pFL-Bench : A Comprehensive Benchmark for Personalized
        # Federated Learning (https://arxiv.org/pdf/2405.17724) for cifar10 dataset we use SGD optimizer
        global_optimizer = torch.optim.SGD(self.global_model.parameters(), lr=self.learning_rate, momentum=0.9)
        local_optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)
        return {"global": global_optimizer, "local": local_optimizer}

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
        "--use_partitioned_data",
        action="store_true",
        help="Use preprocessed partitioned data for training, validation and testing",
        default=False,
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
    if args.use_partitioned_data:
        log(INFO, "Using preprocessed partitioned data for training, validation and testing")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(INFO, f"Device to be used: {device}")
    log(INFO, f"Server Address: {args.server_address}")
    log(INFO, f"Learning Rate: {args.learning_rate}")
    log(INFO, f"Beta: {args.beta}")

    # Set the random seed for reproducibility
    set_all_random_seeds(args.seed)

    # Adding extensive checkpointing for the client
    checkpoint_dir = os.path.join(args.artifact_dir, args.run_name)
    pre_aggregation_best_checkpoint_name = f"pre_aggregation_client_{args.client_number}_best_model.pkl"
    pre_aggregation_last_checkpoint_name = f"pre_aggregation_client_{args.client_number}_last_model.pkl"
    checkpoint_and_state_module = ClientCheckpointAndStateModule(
        pre_aggregation=[
            BestLossTorchModuleCheckpointer(checkpoint_dir, pre_aggregation_best_checkpoint_name),
            LatestTorchModuleCheckpointer(checkpoint_dir, pre_aggregation_last_checkpoint_name),
        ],
    )

    data_path = Path(args.dataset_dir)
    client = CifarDittoClient(
        data_path=data_path,
        metrics=[Accuracy("accuracy")],
        device=device,
        client_number=args.client_number,
        learning_rate=args.learning_rate,
        heterogeneity_level=args.beta,
        checkpoint_and_state_module=checkpoint_and_state_module,
        use_partitioned_data=args.use_partitioned_data,
    )

    fl.client.start_client(server_address=args.server_address, client=client.to_client())
    client.shutdown()
