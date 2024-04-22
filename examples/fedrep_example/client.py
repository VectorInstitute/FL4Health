import argparse
from pathlib import Path
from typing import Dict, Tuple

import flwr as fl
import torch
import torch.nn as nn
from flwr.common.typing import Config
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from examples.models.sequential_split_models import (
    SequentialGlobalFeatureExtractorCifar,
    SequentialLocalPredictionHeadCifar,
)
from fl4health.clients.fedrep_client import FedRepClient
from fl4health.model_bases.fedrep_base import FedRepModel
from fl4health.utils.load_data import load_cifar10_data
from fl4health.utils.metrics import Accuracy
from fl4health.utils.sampler import DirichletLabelBasedSampler


class CifarFedRepClient(FedRepClient):
    def get_data_loaders(self, config: Config) -> Tuple[DataLoader, DataLoader]:
        batch_size = self.narrow_config_type(config, "batch_size", int)
        sample_percentage = self.narrow_config_type(config, "sample_percentage", float)
        beta = self.narrow_config_type(config, "beta", float)
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

    def get_optimizer(self, config: Config) -> Dict[str, Optimizer]:
        # We have two optimizers that are used for the head and representation optimization stages respectively
        assert isinstance(self.model, FedRepModel)
        representation_optimizer = torch.optim.AdamW(self.model.base_module.parameters(), lr=0.001)
        head_optimizer = torch.optim.AdamW(self.model.head_module.parameters(), lr=0.001)
        return {"representation": representation_optimizer, "head": head_optimizer}

    def get_criterion(self, config: Config) -> _Loss:
        return torch.nn.CrossEntropyLoss()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FL Client Main")
    parser.add_argument("--dataset_dir", action="store", type=str, help="Path to the local dataset")
    args = parser.parse_args()

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.dataset_dir)
    client = CifarFedRepClient(data_dir, [Accuracy("accuracy")], DEVICE)
    fl.client.start_client(server_address="0.0.0.0:8080", client=client.to_client())
    client.shutdown()
