import os
from collections import OrderedDict
from logging import INFO
from pathlib import Path
from typing import Dict, Tuple

import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from flwr.common.logger import log
from flwr.common.typing import (
    Config,
    EvaluateRes,
    FitRes,
    GetParametersRes,
    Parameters,
)
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10


class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def load_data(data_dir: Path) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
    """Load CIFAR-10 (training and validation set)."""
    log(INFO, f"Data directory: {str(data_dir)}")
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    training_set = CIFAR10(
        str(data_dir), train=True, download=True, transform=transform
    )
    validation_set = CIFAR10(
        str(data_dir), train=False, download=True, transform=transform
    )
    train_loader = DataLoader(training_set, batch_size=32, shuffle=True)
    validation_loader = DataLoader(validation_set, batch_size=32)
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
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
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
        log(
            INFO,
            f"Epoch: {epoch}, Client Training Loss: {running_loss/n_batches},"
            f"Client Training Accuracy: {accuracy}",
        )
        running_loss = 0.0
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
    log(
        INFO,
        f"Client Validation Loss: {loss/n_batches},"
        f"Client Validation Accuracy: {accuracy}",
    )
    return loss / n_batches, accuracy


class CifarClient(fl.client.NumPyClient):
    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        validation_loader: torch.utils.data.DataLoader,
        num_examples: Dict,
        device: torch.device,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.num_examples = num_examples
        self.device = device

    def get_parameters(self, config: Config) -> GetParametersRes:
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def set_parameters(self, parameters: Parameters, config: Config) -> None:
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters: Parameters, config: Config) -> FitRes:
        self.set_parameters(parameters, config)
        accuracy = train(
            self.model, self.train_loader, epochs=3, device=self.device
        )
        return (
            self.get_parameters(config),
            num_examples["train_set"],
            {"accuracy": accuracy},
        )

    def evaluate(self, parameters: Parameters, config: Config) -> EvaluateRes:
        self.set_parameters(parameters, config)
        loss, accuracy = validate(net, validation_loader, device=self.device)
        return (
            float(loss),
            num_examples["validation_set"],
            {"accuracy": accuracy},
        )


if __name__ == "__main__":
    # Load model and data
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = Net().to(DEVICE)
    train_loader, validation_loader, num_examples = load_data(
        Path(os.path.join(os.getcwd(), "cifar_data/"))
    )
    client = CifarClient(
        net, train_loader, validation_loader, num_examples, DEVICE
    )
    fl.client.start_numpy_client(server_address="0.0.0.0:8080", client=client)
