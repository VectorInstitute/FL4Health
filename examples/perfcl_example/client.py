import argparse
from collections.abc import Sequence, Set
from pathlib import Path

import flwr as fl
import torch
from flwr.common.typing import Config
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from examples.models.parallel_split_cnn import GlobalCnn, LocalCnn, ParallelSplitHeadClassifier
from fl4health.clients.perfcl_client import PerFclClient
from fl4health.metrics import Accuracy
from fl4health.metrics.base_metrics import Metric
from fl4health.model_bases.parallel_split_models import ParallelFeatureJoinMode
from fl4health.model_bases.perfcl_base import PerFclModel
from fl4health.utils.config import narrow_dict_type
from fl4health.utils.load_data import load_mnist_data
from fl4health.utils.sampler import MinorityLabelBasedSampler


class MnistPerFclClient(PerFclClient):
    def __init__(
        self,
        data_path: Path,
        metrics: Sequence[Metric],
        device: torch.device,
        minority_numbers: Set[int],
    ) -> None:
        super().__init__(
            data_path=data_path,
            metrics=metrics,
            device=device,
            global_feature_contrastive_loss_weight=1.0,
            local_feature_contrastive_loss_weight=1.0,
        )
        self.minority_numbers = minority_numbers

    def get_data_loaders(self, config: Config) -> tuple[DataLoader, DataLoader]:
        batch_size = narrow_dict_type(config, "batch_size", int)
        downsample_percentage = narrow_dict_type(config, "downsampling_ratio", float)
        sampler = MinorityLabelBasedSampler(list(range(10)), downsample_percentage, self.minority_numbers)
        train_loader, val_loader, _ = load_mnist_data(self.data_path, batch_size, sampler)
        return train_loader, val_loader

    def get_model(self, config: Config) -> nn.Module:
        model: nn.Module = PerFclModel(
            LocalCnn(), GlobalCnn(), ParallelSplitHeadClassifier(ParallelFeatureJoinMode.CONCATENATE)
        ).to(self.device)
        return model

    def get_optimizer(self, config: Config) -> Optimizer:
        return torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

    def get_criterion(self, config: Config) -> _Loss:
        return torch.nn.CrossEntropyLoss()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FL Client Main")
    parser.add_argument("--dataset_path", action="store", type=str, help="Path to the local dataset")
    parser.add_argument(
        "--minority_numbers", default=[], nargs="*", help="MNIST numbers to be in the minority for the current client"
    )
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_path = Path(args.dataset_path)
    minority_numbers = {int(number) for number in args.minority_numbers}
    client = MnistPerFclClient(data_path, [Accuracy("accuracy")], device, minority_numbers)
    fl.client.start_client(server_address="0.0.0.0:8080", client=client.to_client())
    client.shutdown()
