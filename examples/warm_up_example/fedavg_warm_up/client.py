import argparse
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

from examples.models.cnn_model import MnistNet
from fl4health.checkpointing.checkpointer import LatestTorchModuleCheckpointer
from fl4health.checkpointing.client_module import ClientCheckpointAndStateModule
from fl4health.clients.basic_client import BasicClient
from fl4health.metrics import Accuracy
from fl4health.metrics.base_metrics import Metric
from fl4health.utils.config import narrow_dict_type
from fl4health.utils.load_data import load_mnist_data
from fl4health.utils.random import set_all_random_seeds
from fl4health.utils.sampler import DirichletLabelBasedSampler


class MnistFedAvgClient(BasicClient):
    def __init__(
        self,
        data_path: Path,
        metrics: Sequence[Metric],
        device: torch.device,
        checkpoint_dir: str,
        client_name: str,
    ) -> None:
        # Checkpointing is crucial for the warm up process
        checkpoint_name = f"client_{client_name}_latest_model.pkl"
        post_aggregation_checkpointer = LatestTorchModuleCheckpointer(checkpoint_dir, checkpoint_name)
        checkpoint_and_state_module = ClientCheckpointAndStateModule(post_aggregation=post_aggregation_checkpointer)

        super().__init__(
            data_path=data_path,
            metrics=metrics,
            device=device,
            checkpoint_and_state_module=checkpoint_and_state_module,
            client_name=client_name,
        )

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
        "--checkpoint_dir",
        action="store",
        type=str,
        help="Path to the directory where the checkpoints are stored",
        required=True,
    )
    parser.add_argument(
        "--client_name",
        action="store",
        type=str,
        help="Name for the client, this will also be used to set the checkpoint name",
        required=True,
    )
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_path = Path(args.dataset_path)
    log(INFO, f"Device to be used: {device}")
    log(INFO, f"Server Address: {args.server_address}")

    # Set the random seed for reproducibility
    set_all_random_seeds(args.seed)

    # Start the client
    client = MnistFedAvgClient(
        data_path, [Accuracy()], device, checkpoint_dir=args.checkpoint_dir, client_name=args.client_name
    )
    fl.client.start_client(server_address=args.server_address, client=client.to_client())

    # Shutdown the client gracefully
    client.shutdown()
