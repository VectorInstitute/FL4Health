import argparse
from logging import INFO
from pathlib import Path
from typing import Dict, Tuple

import flwr as fl
import torch
import torch.nn as nn
from flwr.common.logger import log
from flwr.common.typing import Config, NDArrays, Scalar
from torch.utils.data import DataLoader

from examples.dp_fed_examples.client_level_dp_weighted.data import load_data
from examples.models.logistic_regression import LogisticRegression
from fl4health.clients.clipping_client import NumpyClippingClient
from fl4health.parameter_exchange.full_exchanger import FullParameterExchanger


def train(net: nn.Module, train_loader: DataLoader, epochs: int, device: torch.device = torch.device("cpu")) -> float:
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, weight_decay=1e-4)

    for epoch in range(epochs):
        correct, total, running_loss = 0, 0, 0.0
        n_batches = len(train_loader)
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            preds = net(features)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            predicted = preds.data >= 0.5

            total += labels.size(0)
            correct += (predicted.int() == labels.int()).sum().item()

        accuracy = correct / total
        # Local client logging.
        log(
            INFO,
            f"Epoch: {epoch}, Client Training Loss: {running_loss/n_batches},"
            f" Client Training Accuracy: {accuracy}",
        )
    return accuracy


def validate(
    net: nn.Module,
    validation_loader: DataLoader,
    device: torch.device = torch.device("cpu"),
) -> Tuple[float, float]:
    """Validate the network on the entire validation set."""
    criterion = torch.nn.BCELoss()
    correct, total, loss = 0, 0, 0.0
    with torch.no_grad():
        n_batches = len(validation_loader)
        for features, labels in validation_loader:
            features, labels = features.to(device), labels.to(device)
            preds = net(features)
            loss += criterion(preds, labels).item()
            predicted = preds.data >= 0.5
            total += labels.size(0)
            correct += (predicted.int() == labels.int()).sum().item()
    accuracy = correct / total
    # Local client logging.
    log(INFO, f"Client Validation Loss: {loss/n_batches} Client Validation Accuracy: {accuracy}")
    return loss / n_batches, accuracy


class HospitalClient(NumpyClippingClient):
    def __init__(self, data_path: Path, device: torch.device) -> None:
        super().__init__(data_path, device)
        self.model = LogisticRegression(input_dim=31, output_dim=1).to(self.device)
        self.parameter_exchanger = FullParameterExchanger()

    def setup_client(self, config: Config) -> None:
        super().setup_client(config)
        self.batch_size = self.narrow_config_type(config, "batch_size", int)
        self.local_epochs = self.narrow_config_type(config, "local_epochs", int)
        # Server sets clipping strategy and scaler
        self.adaptive_clipping = self.narrow_config_type(config, "adaptive_clipping", bool)
        self.scaler_bytes = self.narrow_config_type(config, "scaler", bytes)

        train_loader, validation_loader, num_examples = load_data(self.data_path, self.batch_size, self.scaler_bytes)

        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.num_examples = num_examples

    def fit(self, parameters: NDArrays, config: Config) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        if not self.initialized:
            self.setup_client(config)

        self.set_parameters(parameters, config)
        accuracy = train(
            self.model,
            self.train_loader,
            self.local_epochs,
            self.device,
        )
        # FitRes should contain local parameters, number of examples on client, and a dictionary holding metrics
        # calculation results.
        return (
            self.get_parameters(config),
            self.num_examples["train_set"],
            {"accuracy": accuracy},
        )

    def evaluate(self, parameters: NDArrays, config: Config) -> Tuple[float, int, Dict[str, Scalar]]:
        if not self.initialized:
            self.setup_client(config)
        self.set_parameters(parameters, config)
        loss, accuracy = validate(self.model, self.validation_loader, device=self.device)
        # EvaluateRes should return the loss, number of examples on client, and a dictionary holding metrics
        # calculation results.
        return (
            loss,
            self.num_examples["validation_set"],
            {"accuracy": accuracy},
        )

    def get_properties(self, config: Config) -> Dict[str, Scalar]:
        """
        Return properties of client.
        First initializes the client because this is called prior to the first
        federated learning round.
        """
        self.setup_client(config)
        return {"num_samples": self.num_examples["train_set"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FL Client Main")
    parser.add_argument("--dataset_path", action="store", type=str, help="Path to the local dataset")
    args = parser.parse_args()

    # Load model and data
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_path = Path(args.dataset_path)
    client = HospitalClient(data_path, DEVICE)
    fl.client.start_numpy_client(server_address="0.0.0.0:8080", client=client)
