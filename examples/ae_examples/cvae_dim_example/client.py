import argparse
from collections.abc import Sequence
from pathlib import Path

import flwr as fl
import torch
from flwr.common.typing import Config
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision import transforms

from examples.models.mnist_model import MnistNet
from fl4health.clients.basic_client import BasicClient
from fl4health.metrics import Accuracy
from fl4health.metrics.base_metrics import Metric
from fl4health.preprocessing.autoencoders.dim_reduction import CvaeFixedConditionProcessor
from fl4health.utils.config import narrow_dict_type
from fl4health.utils.load_data import ToNumpy, load_mnist_data
from fl4health.utils.random import set_all_random_seeds
from fl4health.utils.sampler import DirichletLabelBasedSampler


class CvaeDimClient(BasicClient):
    def __init__(self, data_path: Path, metrics: Sequence[Metric], device: torch.device, condition: torch.Tensor):
        super().__init__(data_path, metrics, device)
        self.condition = condition

    def get_data_loaders(self, config: Config) -> tuple[DataLoader, DataLoader]:
        batch_size = narrow_dict_type(config, "batch_size", int)
        cvae_model_path = Path(narrow_dict_type(config, "cvae_model_path", str))
        sampler = DirichletLabelBasedSampler(list(range(10)), sample_percentage=0.75, beta=100)
        transform = transforms.Compose([ToNumpy(), transforms.ToTensor(), transforms.Lambda(torch.flatten)])
        # CvaeFixedConditionProcessor is added to the data transform pipeline to encode the data samples
        train_loader, val_loader, _ = load_mnist_data(
            data_dir=self.data_path,
            batch_size=batch_size,
            sampler=sampler,
            transform=transforms.Compose([transform, CvaeFixedConditionProcessor(cvae_model_path, self.condition)]),
        )
        return train_loader, val_loader

    def get_criterion(self, config: Config) -> _Loss:
        return torch.nn.CrossEntropyLoss()

    def get_optimizer(self, config: Config) -> Optimizer:
        return torch.optim.Adam(self.model.parameters(), lr=0.001)

    def get_model(self, config: Config) -> nn.Module:
        latent_dim = narrow_dict_type(config, "latent_dim", int)
        # Dimensionality reduction reduces the size of inputs to the size of cat(mu, logvar).
        return MnistNet(latent_dim * 2).to(self.device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FL Client Main")
    parser.add_argument("--dataset_path", action="store", type=str, help="Path to the local dataset")
    parser.add_argument("--condition", action="store", type=int, help="Client ID used for CVAE")
    parser.add_argument(
        "--num_conditions",
        action="store",
        type=int,
        help="Total number of conditions to create the condition vector. Total number of clients in this example.",
    )
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_path = Path(args.dataset_path)
    set_all_random_seeds(42)
    # Creating the condition vector used for training this CVAE.
    condition_vector = torch.nn.functional.one_hot(torch.tensor(args.condition), num_classes=args.num_conditions)
    client = CvaeDimClient(data_path, [Accuracy("accuracy")], device, condition_vector)
    fl.client.start_client(server_address="0.0.0.0:8080", client=client.to_client())
    client.shutdown()
