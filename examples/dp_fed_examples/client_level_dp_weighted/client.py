import argparse
from pathlib import Path
from typing import Tuple

import flwr as fl
import torch
import torch.nn as nn
from flwr.common.typing import Config
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from examples.dp_fed_examples.client_level_dp_weighted.data import load_data
from examples.models.logistic_regression import LogisticRegression
from fl4health.clients.clipping_client import NumpyClippingClient
from fl4health.utils.metrics import Accuracy


class HospitalClient(NumpyClippingClient):
    def get_model(self, config: Config) -> nn.Module:
        return LogisticRegression(input_dim=31, output_dim=1).to(self.device)

    def get_data_loaders(self, config: Config) -> Tuple[DataLoader, DataLoader]:
        batch_size = self.narrow_config_type(config, "batch_size", int)
        scaler_bytes = self.narrow_config_type(config, "scaler", bytes)
        train_loader, val_loader, _ = load_data(self.data_path, batch_size, scaler_bytes)
        return train_loader, val_loader

    def get_optimizer(self, config: Config) -> Optimizer:
        return torch.optim.SGD(self.model.parameters(), lr=0.01, weight_decay=1e-4)

    def get_criterion(self, config: Config) -> _Loss:
        return torch.nn.BCELoss()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FL Client Main")
    parser.add_argument("--dataset_path", action="store", type=str, help="Path to the local dataset")
    args = parser.parse_args()

    # Load model and data
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_path = Path(args.dataset_path)
    client = HospitalClient(data_path, [Accuracy("accuracy")], DEVICE)
    fl.client.start_numpy_client(server_address="0.0.0.0:8080", client=client)
    client.shutdown()
