import argparse
from pathlib import Path

import flwr as fl
import torch
import torch.nn as nn
from flwr.common.typing import Config
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader, TensorDataset

# from examples.models.cnn_model import Net
from fl4health.clients.basic_client import BasicClient
from fl4health.utils.config import narrow_dict_type

# from fl4health.utils.load_data import load_cifar10_data, load_cifar10_test_data
from fl4health.utils.metrics import Accuracy

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def load_fraud_data(file_path, batch_size):
    """
    Input:
        file_path : file path of the main train file
        batch_size: enough said

    Output:
        train_data_loader
        val_data_loader
    """
    filename = Path(file_path) / "fraud_train.csv"
    df = pd.read_csv(filename, parse_dates=["timestamp"])

    # split label from df
    label = df["fraud_label"]
    df.drop(columns=["timestamp", "fraud_label"], inplace=True)

    x_train, x_val, y_train, y_val = train_test_split(df, label, test_size=0.2)

    ## Train dataloader:
    train_set = TensorDataset(
        torch.tensor(x_train.values, dtype=torch.float32),
        torch.tensor(y_train.values, dtype=torch.float32).reshape(-1, 1),
    )
    train_loader = DataLoader(train_set, batch_size=batch_size)

    ## val dataloader:
    val_set = TensorDataset(
        torch.tensor(x_val.values, dtype=torch.float32), torch.tensor(y_val.values, dtype=torch.float32).reshape(-1, 1)
    )
    val_loader = DataLoader(val_set, batch_size=batch_size)

    return train_loader, val_loader


def load_test_fraud_data(file_path, batch_size):
    filename = Path(file_path) / "fraud_test.csv"
    df = pd.read_csv(filename, parse_dates=["timestamp"])

    # split label from df
    label = df["fraud_label"]
    df.drop(columns=["timestamp", "fraud_label"], inplace=True)

    ## Test dataloader:
    test_set = TensorDataset(
        torch.tensor(df.values, dtype=torch.float32),
        torch.tensor(label.values, dtype=torch.float32).reshape(-1, 1),
    )
    test_loader = DataLoader(test_set, batch_size=batch_size)

    return test_loader


class CifarClient(BasicClient):
    def get_data_loaders(self, config: Config) -> tuple[DataLoader, DataLoader]:
        batch_size = narrow_dict_type(config, "batch_size", int)
        train_loader, val_loader = load_fraud_data(self.data_path, batch_size)
        return train_loader, val_loader

    def get_test_data_loader(self, config: Config) -> DataLoader | None:
        batch_size = narrow_dict_type(config, "batch_size", int)
        test_loader = load_test_fraud_data(self.data_path, batch_size)
        return test_loader

    def get_criterion(self, config: Config) -> _Loss:
        # return nn.BCEWithLogitsLoss() #replaced with BCELoss to use sigmoid layer in model
        return nn.BCELoss()

    def get_optimizer(self, config: Config) -> Optimizer:
        return torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

    def get_model(self, config: Config) -> nn.Module:
        model = nn.Sequential(
            nn.Linear(32, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.1), nn.Linear(32, 1), nn.Sigmoid()
        )
        model.train()
        return model.to(self.device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FL Client Main")
    parser.add_argument("--dataset_path", action="store", type=str, help="Path to the local dataset")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_path = Path(args.dataset_path)
    client = CifarClient(data_path, [Accuracy("accuracy")], device)
    fl.client.start_client(server_address="0.0.0.0:8080", client=client.to_client())
    # fl.client.start_client(server_address="localhost:8080", client=client.to_client())
    client.shutdown()
