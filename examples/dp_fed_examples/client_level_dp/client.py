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

from examples.models.cnn_model import Net
from fl4health.clients.clipping_client import NumpyClippingClient
from fl4health.parameter_exchange.packing_exchanger import ParameterExchangerWithClippingBit
from fl4health.utils.load_data import load_cifar10_data


def train(
    net: nn.Module,
    train_loader: DataLoader,
    epochs: int,
    device: torch.device = torch.device("cpu"),
) -> float:
    """Train the network on the training set."""
    net.train()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(net.parameters(), lr=0.0001)

    for epoch in range(epochs):
        correct, total, running_loss = 0, 0, 0.0
        n_batches = len(train_loader)
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            preds = net(images)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(preds.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

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
    net.eval()
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    with torch.no_grad():
        n_batches = len(validation_loader)
        for images, labels in validation_loader:
            images, labels = images.to(device), labels.to(device)
            preds = net(images)
            loss += criterion(preds, labels).item()
            _, predicted = torch.max(preds.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    # Local client logging.
    log(INFO, f"Client Validation Loss: {loss/n_batches} Client Validation Accuracy: {accuracy}")
    return loss / n_batches, accuracy


class CifarClient(NumpyClippingClient):
    def __init__(self, data_path: Path, device: torch.device) -> None:
        super().__init__(data_path, device)
        self.model = Net().to(self.device)
        self.parameter_exchanger = ParameterExchangerWithClippingBit()

    def setup_client(self, config: Config) -> None:
        super().setup_client(config)
        self.batch_size = self.narrow_config_type(config, "batch_size", int)
        self.local_epochs = self.narrow_config_type(config, "local_epochs", int)
        # Server explicitly sets the clipping strategy
        self.adaptive_clipping = self.narrow_config_type(config, "adaptive_clipping", bool)

        train_loader, validation_loader, num_examples = load_cifar10_data(self.data_path, self.batch_size)

        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.num_examples = num_examples

    def fit(self, parameters: NDArrays, config: Config) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FL Client Main")
    parser.add_argument("--dataset_path", action="store", type=str, help="Path to the local dataset")
    args = parser.parse_args()

    # Load model and data
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_path = Path(args.dataset_path)
    client = CifarClient(data_path, DEVICE)
    fl.client.start_numpy_client(server_address="0.0.0.0:8080", client=client)
