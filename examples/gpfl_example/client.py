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

from examples.models.sequential_split_models import (
    SequentialGlobalFeatureExtractorMnist,
    SequentialLocalPredictionHeadMnist,
)
from fl4health.clients.gpfl_client import GpflClient
from fl4health.metrics import Accuracy
from fl4health.metrics.base_metrics import Metric
from fl4health.model_bases.gpfl_base import GpflModel
from fl4health.reporting import JsonReporter
from fl4health.reporting.base_reporter import BaseReporter
from fl4health.utils.config import narrow_dict_type
from fl4health.utils.load_data import load_mnist_data, load_mnist_test_data
from fl4health.utils.random import set_all_random_seeds


class MnistGpflClient(GpflClient):
    def __init__(
        self,
        data_path: Path,
        metrics: Sequence[Metric],
        device: torch.device,
        reporters: Sequence[BaseReporter] | None = None,
        mu: float = 0.01,
        lam: float = 0.01,
        learning_rate: float = 0.005,
    ) -> None:
        super().__init__(data_path=data_path, metrics=metrics, device=device, reporters=reporters, lam=lam, mu=mu)
        self.learning_rate: float = learning_rate

    def get_data_loaders(self, config: Config) -> tuple[DataLoader, DataLoader]:
        batch_size = narrow_dict_type(config, "batch_size", int)
        train_loader, val_loader, _ = load_mnist_data(self.data_path, batch_size)
        return train_loader, val_loader

    def get_test_data_loader(self, config: Config) -> DataLoader | None:
        batch_size = narrow_dict_type(config, "batch_size", int)
        test_loader, _ = load_mnist_test_data(self.data_path, batch_size)
        return test_loader

    def get_criterion(self, config: Config) -> _Loss:
        return torch.nn.CrossEntropyLoss()

    def get_optimizer(self, config: Config) -> dict[str, Optimizer]:
        return {
            "model": torch.optim.SGD(self.model.gpfl_main_module.parameters(), lr=self.learning_rate),
            "gce": torch.optim.SGD(self.model.gce.embedding.parameters(), lr=self.learning_rate, weight_decay=self.mu),
            "cov": torch.optim.SGD(self.model.cov.parameters(), lr=self.learning_rate, weight_decay=self.mu),
        }

    def get_model(self, config: Config) -> nn.Module:
        model = GpflModel(
            base_module=SequentialGlobalFeatureExtractorMnist(),
            head_module=SequentialLocalPredictionHeadMnist(),
            feature_dim=120,  # This should match the output dimension of the global feature extractor
            num_classes=10,  # Number of classes based on the dataset.
            flatten_features=True,
        )
        return model.to(self.device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPFL Client Main")
    parser.add_argument("--dataset_path", action="store", type=str, help="Path to the local dataset")
    parser.add_argument(
        "--seed",
        action="store",
        type=int,
        help="Seed for the random number generators across python, torch, and numpy",
        required=False,
    )
    parser.add_argument(
        "--learning_rate", action="store", type=float, default=0.005, help="Learning rate used for all the optimizers"
    )
    parser.add_argument(
        "--mu",
        action="store",
        type=float,
        default=0.01,
        help="Mu parameter used as weight decay for GCE and CoV optimizers",
    )
    parser.add_argument(
        "--lambda_parameter",
        action="store",
        type=float,
        default=0.01,
        help="Lambda parameter used to weight magnitude level global loss",
    )
    args = parser.parse_args()
    # Set the random seed for reproducibility
    set_all_random_seeds(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_path = Path(args.dataset_path)
    client = MnistGpflClient(
        data_path,
        [Accuracy("accuracy")],
        device,
        reporters=[JsonReporter()],
        mu=args.mu,
        lam=args.lambda_parameter,
        learning_rate=args.learning_rate,
    )
    fl.client.start_client(server_address="0.0.0.0:8080", client=client.to_client())
    client.shutdown()
