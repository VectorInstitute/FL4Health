import os
from collections import OrderedDict
from logging import INFO, WARNING
from pathlib import Path
from typing import Dict, Tuple

import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from flwr.common.logger import log
from flwr.common.typing import Config, NDArrays, Scalar
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
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
    optimizer: torch.optim.Optimizer,
    epochs: int,
    device: torch.device = torch.device("cpu"),
) -> float:
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()

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
            f"Epoch: {epoch}, Client Training Loss: {running_loss/n_batches}," f"Client Training Accuracy: {accuracy}",
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
    log(
        INFO,
        f"Client Validation Loss: {loss/n_batches}," f"Client Validation Accuracy: {accuracy}",
    )
    return loss / n_batches, accuracy


class CifarClient(fl.client.NumPyClient):
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
    ) -> None:
        self.model = model
        self.device = device
        self.initialized = False
        self.train_loader: DataLoader

    def get_parameters(self, config: Config) -> NDArrays:
        # Determines which weights are sent back to the server for aggregation.
        # Currently sending all of them ordered by state_dict keys
        # NOTE: Order matters, because it is relied upon by set_parameters below
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: NDArrays, config: Config) -> None:
        # Sets the local model parameters transfered from the server. The state_dict is
        # reconstituted because parameters is simply a list of bytes
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def setup_opacus_objects(self) -> None:
        self.optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        # Validate that the model layers are compatible with privacy mechanisms in Opacus and try to replace the layers
        # with compatible ones if necessary.
        errors = ModuleValidator.validate(self.model, strict=False)
        if len(errors) != 0:
            for error in errors:
                log(WARNING, f"Opacus error: {error}")
            self.model = ModuleValidator.fix(self.model)

        # Create DP training objects
        privacy_engine = PrivacyEngine()
        # NOTE: that Opacus make private is NOT idempotent
        self.model, self.optimizer, self.train_loader = privacy_engine.make_private(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=self.train_loader,
            noise_multiplier=self.noise_multiplier,
            max_grad_norm=self.clipping_bound,
        )

    def setup_client(self, config: Config) -> None:
        self.batch_size = config["batch_size"]
        self.local_epochs = config["local_epochs"]
        self.noise_multiplier = config["noise_multiplier"]
        self.clipping_bound = config["clipping_bound"]

        train_loader, validation_loader, num_examples = load_data(
            Path(os.path.join(os.path.dirname(os.getcwd()), "examples", "datasets", "cifar_data")), self.batch_size
        )

        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.num_examples = num_examples
        self.initialized = True
        self.setup_opacus_objects()

    def fit(self, parameters: NDArrays, config: Config) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        if not self.initialized:
            self.setup_client(config)
        self.set_parameters(parameters, config)
        accuracy = train(
            self.model,
            self.train_loader,
            self.optimizer,
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
    # Load model and data
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = Net().to(DEVICE)
    client = CifarClient(net, DEVICE)
    fl.client.start_numpy_client(server_address="0.0.0.0:8080", client=client)
