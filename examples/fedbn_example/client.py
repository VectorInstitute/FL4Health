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

from examples.models.cnn_model import MnistNetWithBnAndFrozen, SkinCancerNet
from fl4health.clients.fedbn_client import FedBnClient
from fl4health.datasets.skin_cancer.load_data import load_skin_cancer_data
from fl4health.metrics import Accuracy
from fl4health.metrics.base_metrics import Metric
from fl4health.parameter_exchange.layer_exchanger import LayerExchangerWithExclusions
from fl4health.parameter_exchange.parameter_exchanger_base import ParameterExchanger
from fl4health.utils.config import narrow_dict_type
from fl4health.utils.load_data import load_mnist_data
from fl4health.utils.sampler import DirichletLabelBasedSampler


class MnistFedBnClient(FedBnClient):
    def get_data_loaders(self, config: Config) -> tuple[DataLoader, DataLoader]:
        batch_size = narrow_dict_type(config, "batch_size", int)
        sampler = DirichletLabelBasedSampler(list(range(10)), sample_percentage=0.75, beta=1)
        train_loader, val_loader, _ = load_mnist_data(self.data_path, batch_size, sampler)
        return train_loader, val_loader

    def get_optimizer(self, config: Config) -> Optimizer:
        return torch.optim.AdamW(self.model.parameters(), lr=0.01)

    def get_criterion(self, config: Config) -> _Loss:
        return torch.nn.CrossEntropyLoss()

    def get_model(self, config: Config) -> nn.Module:
        return MnistNetWithBnAndFrozen(freeze_cnn_layer=False).to(self.device)

    def get_parameter_exchanger(self, config: Config) -> ParameterExchanger:
        assert self.model is not None
        return LayerExchangerWithExclusions(self.model, {nn.BatchNorm2d})


class SkinCancerFedBNClient(FedBnClient):
    def __init__(self, data_path: Path, metrics: Sequence[Metric], device: torch.device, dataset_name: str):
        super().__init__(data_path, metrics, device)
        self.dataset_name = dataset_name

    def get_data_loaders(self, config: Config) -> tuple[DataLoader, DataLoader]:
        batch_size = narrow_dict_type(config, "batch_size", int)
        train_loader, val_loader, _, _ = load_skin_cancer_data(self.data_path, self.dataset_name, batch_size)
        return train_loader, val_loader

    def get_optimizer(self, config: Config) -> Optimizer:
        return torch.optim.AdamW(self.model.parameters(), lr=0.01)

    def get_criterion(self, config: Config) -> _Loss:
        return torch.nn.CrossEntropyLoss()

    def get_model(self, config: Config) -> nn.Module:
        return SkinCancerNet().to(self.device)

    def get_parameter_exchanger(self, config: Config) -> ParameterExchanger:
        assert self.model is not None
        return LayerExchangerWithExclusions(self.model, {nn.BatchNorm2d})


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
        "--dataset_name",
        action="store",
        type=str,
        help="Dataset name (e.g., Barcelona, Rosendahl, Vienna, UFES, Canada for Skin Cancer; \
            'mnist' for MNIST dataset)",
        default="mnist",
    )
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_path = Path(args.dataset_path)
    log(INFO, f"Device to be used: {device}")
    log(INFO, f"Server Address: {args.server_address}")

    if args.dataset_name in ["Barcelona", "Rosendahl", "Vienna", "UFES", "Canada"]:
        client: FedBnClient = SkinCancerFedBNClient(data_path, [Accuracy()], device, args.dataset_name)
    elif args.dataset_name == "mnist":
        client = MnistFedBnClient(data_path, [Accuracy()], device)
    else:
        raise ValueError(
            "Unsupported dataset name. Please choose from 'Barcelona', 'Rosendahl', \
            'Vienna', 'UFES', 'Canada', or 'mnist'."
        )

    fl.client.start_client(server_address=args.server_address, client=client.to_client())

    # Shutdown the client gracefully
    client.shutdown()
