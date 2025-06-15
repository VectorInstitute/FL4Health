import argparse
from collections.abc import Sequence
from logging import INFO
from pathlib import Path

import flwr as fl
import torch
from flwr.common.logger import log
from flwr.common.typing import Config, NDArrays
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from examples.models.cnn_model import MnistNet
from fl4health.clients.fed_prox_client import FedProxClient
from fl4health.metrics import Accuracy
from fl4health.metrics.base_metrics import Metric
from fl4health.preprocessing.warmed_up_module import WarmedUpModule
from fl4health.utils.config import narrow_dict_type
from fl4health.utils.load_data import load_mnist_data
from fl4health.utils.random import set_all_random_seeds
from fl4health.utils.sampler import DirichletLabelBasedSampler


class MnistFedProxClient(FedProxClient):
    def __init__(
        self,
        data_path: Path,
        metrics: Sequence[Metric],
        device: torch.device,
        pretrained_model_path: Path,
        weights_mapping_path: Path | None,
    ) -> None:
        super().__init__(
            data_path=data_path,
            metrics=metrics,
            device=device,
        )
        # Load the warmed up module
        self.warmed_up_module = WarmedUpModule(
            pretrained_model_path=pretrained_model_path,
            weights_mapping_path=weights_mapping_path,
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

    # Overriding the initialize_all_model_weights function is required to load the pretrained model
    def initialize_all_model_weights(self, parameters: NDArrays, config: Config) -> None:
        super().initialize_all_model_weights(parameters, config)
        # Load the pretrained model
        self.warmed_up_module.load_from_pretrained(self.model)


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
        "--pretrained_model_path",
        action="store",
        type=str,
        help="Path to the pretrained model",
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

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_path = Path(args.dataset_path)
    pretrained_model_path = Path(args.pretrained_model_path)
    weights_mapping_path = Path(args.weights_mapping_path) if args.weights_mapping_path else None
    log(INFO, f"Device to be used: {device}")
    log(INFO, f"Server Address: {args.server_address}")

    # Set the random seed for reproducibility
    set_all_random_seeds(args.seed)

    # Start the client
    client = MnistFedProxClient(
        data_path,
        [Accuracy()],
        device,
        pretrained_model_path,
        weights_mapping_path,
    )
    fl.client.start_client(server_address=args.server_address, client=client.to_client())

    # Shutdown the client gracefully
    client.shutdown()
