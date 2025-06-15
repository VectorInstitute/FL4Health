import argparse
from pathlib import Path

import flwr as fl
import torch
from flwr.common.typing import Config
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from examples.models.sequential_split_models import (
    SequentialGlobalFeatureExtractorCifar,
    SequentialLocalPredictionHeadCifar,
)
from fl4health.clients.fedrep_client import FedRepClient
from fl4health.metrics import Accuracy
from fl4health.model_bases.fedrep_base import FedRepModel
from fl4health.utils.config import narrow_dict_type
from fl4health.utils.load_data import load_cifar10_data
from fl4health.utils.sampler import DirichletLabelBasedSampler


class CifarFedRepClient(FedRepClient):
    def get_data_loaders(self, config: Config) -> tuple[DataLoader, DataLoader]:
        batch_size = narrow_dict_type(config, "batch_size", int)
        sample_percentage = narrow_dict_type(config, "sample_percentage", float)
        beta = narrow_dict_type(config, "beta", float)
        assert beta > 0 and 0 < sample_percentage < 1
        sampler = DirichletLabelBasedSampler(list(range(10)), sample_percentage=sample_percentage, beta=beta)
        train_loader, val_loader, _ = load_cifar10_data(self.data_path, batch_size, sampler=sampler)
        return train_loader, val_loader

    def get_model(self, config: Config) -> nn.Module:
        model: nn.Module = FedRepModel(
            base_module=SequentialGlobalFeatureExtractorCifar(),
            head_module=SequentialLocalPredictionHeadCifar(),
        ).to(self.device)
        return model

    def get_optimizer(self, config: Config) -> dict[str, Optimizer]:
        # We have two optimizers that are used for the head and representation optimization stages respectively
        assert isinstance(self.model, FedRepModel)
        representation_optimizer = torch.optim.AdamW(self.model.base_module.parameters(), lr=0.001)
        head_optimizer = torch.optim.AdamW(self.model.head_module.parameters(), lr=0.001)
        return {"representation": representation_optimizer, "head": head_optimizer}

    def get_criterion(self, config: Config) -> _Loss:
        return torch.nn.CrossEntropyLoss()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FL Client Main")
    parser.add_argument("--dataset_path", action="store", type=str, help="Path to the local dataset")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.dataset_path)
    client = CifarFedRepClient(data_dir, [Accuracy("accuracy")], device)
    fl.client.start_client(server_address="0.0.0.0:8080", client=client.to_client())
    client.shutdown()
