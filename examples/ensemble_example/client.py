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

from examples.models.ensemble_cnn import ConfigurableMnistNet
from fl4health.clients.ensemble_client import EnsembleClient
from fl4health.model_bases.ensemble_base import EnsembleModel
from fl4health.utils.load_data import load_mnist_data
from fl4health.utils.metrics import Accuracy
from fl4health.utils.sampler import DirichletLabelBasedSampler


class MnistEnsembleClient(EnsembleClient):
    def get_data_loaders(self, config: Config) -> Tuple[DataLoader, DataLoader]:
        batch_size = self.narrow_config_type(config, "batch_size", int)
        sampler = DirichletLabelBasedSampler(list(range(10)), sample_percentage=0.75)
        train_loader, val_loader, _ = load_mnist_data(self.data_path, batch_size, sampler)
        return train_loader, val_loader

    def get_model(self, config: Config) -> nn.Module:
        ensemble_models: Dict[str, nn.Module] = {
            "ensemble-model-0": ConfigurableMnistNet(out_channel_mult=1).to(self.device),
            "ensemble-model-1": ConfigurableMnistNet(out_channel_mult=2).to(self.device),
            "ensemble-model-2": ConfigurableMnistNet(out_channel_mult=3).to(self.device),
        }
        return EnsembleModel(ensemble_models)

    def get_optimizer(self, config: Config) -> Dict[str, Optimizer]:
        ensemble_optimizers: Dict[str, torch.optim.Optimizer] = {
            "ensemble-model-0": torch.optim.AdamW(self.model.models["ensemble-model-0"].parameters(), lr=0.01),
            "ensemble-model-1": torch.optim.AdamW(self.model.models["ensemble-model-1"].parameters(), lr=0.01),
            "ensemble-model-2": torch.optim.AdamW(self.model.models["ensemble-model-2"].parameters(), lr=0.01),
        }
        return ensemble_optimizers

    def get_criterion(self, config: Config) -> _Loss:
        return torch.nn.CrossEntropyLoss()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FL Client Main")
    parser.add_argument("--dataset_path", action="store", type=str, help="Path to the local dataset")

    args = parser.parse_args()

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_path = Path(args.dataset_path)

    client = MnistEnsembleClient(data_path, [Accuracy()], DEVICE)
    fl.client.start_numpy_client(server_address="0.0.0.0:8080", client=client)
