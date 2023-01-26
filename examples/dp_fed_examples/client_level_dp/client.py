import argparse
from logging import INFO
from pathlib import Path
from typing import Dict, Tuple

import flwr as fl
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from flwr.common.logger import log
from flwr.common.typing import Config, NDArrays, Scalar
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

from examples.models.cnn_model import Net
from fl4health.clients.clipping_client import NumpyClippingClient
from fl4health.parameter_exchange.full_exchanger import FullParameterExchanger


def load_data(data_dir: Path, batch_size: int) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
    """Load CIFAR-10 (training and validation set)."""
    log(INFO, f"Data directory: {str(data_dir)}")
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    training_set = CIFAR10(str(data_dir), train=True, download=True, transform=transform)
    validation_set = CIFAR10(str(data_dir), train=False, download=True, transform=transform)
    train_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_set, batch_size=batch_size)
    num_examples = {
        "train_set": len(training_set),
        "validation_set": len(validation_set),
    }
    return train_loader, validation_loader, num_examples


def train(
    net: nn.Module,
    train_loader: DataLoader,
    epochs: int,
    device: torch.device = torch.device("cpu"),
) -> float:
    """Train the network on the training set."""
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
    def setup_client(self, config: Config) -> None:

        self.batch_size = self.narrow_config_type(config, "batch_size", int)
        self.local_epochs = self.narrow_config_type(config, "local_epochs", int)
        self.adaptive_clipping = self.narrow_config_type(config, "adaptive_clipping", bool)

        train_loader, validation_loader, num_examples = load_data(self.data_path, self.batch_size)

        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.num_examples = num_examples

        self.model = Net().to(self.device)
        self.parameter_exchanger = FullParameterExchanger()
        self.initialized = True

    def fit(self, parameters: NDArrays, config: Config) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        # Expectation is that the last entry in the parameters NDArrays is a clipping bound
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
        # Expectation is that the last entry in the parameters NDArrays is a clipping bound (even if it isn't used
        # for evaluation)
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
