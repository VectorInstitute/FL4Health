import argparse
import os
from logging import INFO
from pathlib import Path
from typing import Optional, Sequence

import flwr as fl
import torch
import torch.nn as nn
from flwr.common.logger import log
from flwr.common.typing import Config, Tuple
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from examples.models.cnn_model import MnistNet
from fl4health.clients.fed_prox_client import FedProxClient
from fl4health.preprocessing.warmed_up_module import WarmedUpModule
from fl4health.utils.load_data import load_mnist_data
from fl4health.utils.metrics import Accuracy, Metric
from fl4health.utils.random import set_all_random_seeds
from fl4health.utils.sampler import DirichletLabelBasedSampler


class MnistFedProxClient(FedProxClient):
    def __init__(
        self,
        data_path: Path,
        metrics: Sequence[Metric],
        device: torch.device,
        client_number: int,
        pretrained_model_dir: Path,
        weights_mapping_path: Optional[Path],
    ) -> None:
        super().__init__(
            data_path=data_path,
            metrics=metrics,
            device=device,
        )

        # Load the warmed up module
        pretrained_model_name = f"client_{client_number}_latest_model.pkl"
        self.warmed_up_module = WarmedUpModule(
            pretrained_model_path=Path(os.path.join(pretrained_model_dir, pretrained_model_name)),
            weights_mapping_path=weights_mapping_path,
        )

    def get_data_loaders(self, config: Config) -> Tuple[DataLoader, DataLoader]:
        sampler = DirichletLabelBasedSampler(list(range(10)), sample_percentage=0.75, beta=1)
        batch_size = self.narrow_config_type(config, "batch_size", int)
        train_loader, val_loader, _ = load_mnist_data(self.data_path, batch_size, sampler)
        return train_loader, val_loader

    def get_model(self, config: Config) -> nn.Module:

        # Load the pretrained model
        model = self.warmed_up_module.load_from_pretrained(MnistNet()).to(self.device)

        # To not overwrite the pretrained model with server model weights
        self.model_weights_initialized = True

        return model

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
        help="Seed for the random number generator",
        required=False,
    )
    parser.add_argument(
        "--pretrained_model_dir",
        action="store",
        type=str,
        help="Path to the pretrained model",
        required=False,
    )
    parser.add_argument(
        "--client_number",
        action="store",
        type=int,
        help="Number of the client for the loading of the pretrained model",
        required=True,
    )
    parser.add_argument(
        "--weights_mapping_path",
        action="store",
        type=str,
        help="Path to the weights mapping file",
        required=False,
    )
    args = parser.parse_args()

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_path = Path(args.dataset_path)
    log(INFO, f"Device to be used: {DEVICE}")
    log(INFO, f"Server Address: {args.server_address}")

    # Set the random seed for reproducibility
    set_all_random_seeds(args.seed)

    # Start the client
    client = MnistFedProxClient(
        data_path,
        [Accuracy()],
        DEVICE,
        args.client_number,
        Path(args.pretrained_model_dir),
        Path(args.weights_mapping_path),
    )
    fl.client.start_numpy_client(server_address=args.server_address, client=client)

    # Shutdown the client gracefully
    client.shutdown()