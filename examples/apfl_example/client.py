import argparse
from logging import INFO
from pathlib import Path
from typing import Dict, Set, Tuple

import flwr as fl
import torch
import torch.nn as nn
from flwr.common.logger import log
from flwr.common.typing import Config, NDArrays, Scalar
from torch.utils.data import DataLoader

from examples.datasets.dataset_utils import load_mnist_data
from examples.models.cnn_model import MNISTNet
from fl4health.clients.numpy_fl_client import NumpyFlClient
from fl4health.model_bases.apfl_base import APFLCriterion, APFLModule, APFLOptimizer
from fl4health.parameter_exchange.layer_exchanger import FixedLayerExchanger


def train(
    net: nn.Module,
    train_loader: DataLoader,
    epochs: int,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Scalar]:
    """Train the network on the training set."""
    criterion = APFLCriterion(torch.nn.CrossEntropyLoss())
    optimizer = APFLOptimizer(net, torch.optim.SGD, lr=0.001, momentum=0.9)

    correct = {"global": 0, "local": 0, "personalized": 0.0}

    for epoch in range(epochs):
        running_losses = {"global": 0.0, "local": 0.0, "personalized": 0.0}
        total = 0
        n_batches = len(train_loader)
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(images)
            losses = criterion(outputs, labels)
            losses.backward()
            optimizer.step()
            total += labels.size(0)
            running_losses = {key: running_loss + losses[key].item() for key, running_loss in running_losses.items()}
            preds = {key: torch.max(output.data, 1)[1] for key, output in outputs.items()}
            correct = {key: count + (preds[key] == labels).sum().item() for key, count in correct.items()}

        running_losses = {key: running_loss / n_batches for key, running_loss in running_losses.items()}
        accuracy: Dict[str, Scalar] = {f"{key}_accuracy": count / total for key, count in correct.items()}

        # Local client logging.
        log(
            INFO,
            f"Epoch: {epoch}, Client Training Loss: {running_losses},"
            f"Client Training Accuracy: {accuracy}, alpha:{net.alpha}",
        )

    return accuracy


def validate(
    net: nn.Module,
    validation_loader: DataLoader,
    device: torch.device = torch.device("cpu"),
) -> Tuple[float, Dict[str, Scalar]]:
    """Validate the network on the entire validation set."""
    criterion = APFLCriterion(torch.nn.CrossEntropyLoss())

    correct = {"global": 0, "local": 0, "personalized": 0.0}
    running_losses = {"global": 0.0, "local": 0.0, "personalized": 0.0}
    total = 0
    n_batches = len(validation_loader)
    with torch.no_grad():
        for images, labels in validation_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            losses = criterion(outputs, labels)
            total += labels.size(0)
            running_losses = {key: running_loss + losses[key].item() for key, running_loss in running_losses.items()}
            preds = {key: torch.max(output.data, 1)[1] for key, output in outputs.items()}
            correct = {key: count + (preds[key] == labels).sum().item() for key, count in correct.items()}

    running_losses = {key: running_loss / n_batches for key, running_loss in running_losses.items()}
    accuracy: Dict[str, Scalar] = {f"{key}_accuracy": count / total for key, count in correct.items()}

    # Local client logging.
    log(
        INFO,
        f"Client Validation Loss: {running_losses}," f"Client Validation Accuracy: {accuracy}",
    )
    return running_losses["global"], accuracy


class MnistAPFLClient(NumpyFlClient):
    def __init__(
        self,
        data_path: Path,
        minority_numbers: Set[int],
        device: torch.device,
    ) -> None:
        super().__init__(data_path, device)
        self.minority_numbers = minority_numbers
        self.model = APFLModule(MNISTNet()).to(self.device)
        self.parameter_exchanger = FixedLayerExchanger(self.model.layers_to_exchange())

    def fit(self, parameters: NDArrays, config: Config) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        if not self.initialized:
            self.setup_client(config)

        self.set_parameters(parameters, config)
        local_epochs = self.narrow_config_type(config, "local_epochs", int)
        accuracy = train(self.model, self.train_loader, epochs=local_epochs, device=self.device)
        # FitRes should contain local parameters, number of examples on client, and a dictionary holding metrics
        # calculation results.
        return (
            self.get_parameters(config),
            self.num_examples["train_set"],
            accuracy,
        )

    def evaluate(self, parameters: NDArrays, config: Config) -> Tuple[float, int, Dict[str, Scalar]]:
        self.set_parameters(parameters, config)
        loss, accuracy = validate(self.model, self.validation_loader, device=self.device)
        # EvaluateRes should return the loss, number of examples on client, and a dictionary holding metrics
        # calculation results.
        return (
            loss,
            self.num_examples["validation_set"],
            accuracy,
        )

    def setup_client(self, config: Config) -> None:
        super().setup_client(config)
        batch_size = self.narrow_config_type(config, "batch_size", int)
        downsample_percentage = self.narrow_config_type(config, "downsampling_ratio", float)

        train_loader, validation_loader, num_examples = load_mnist_data(
            self.data_path, batch_size, downsample_percentage, self.minority_numbers
        )

        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.num_examples = num_examples


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FL Client Main")
    parser.add_argument("--dataset_path", action="store", type=str, help="Path to the local dataset")
    parser.add_argument(
        "--minority_numbers", default=[], nargs="*", help="MNIST numbers to be in the minority for the current client"
    )
    args = parser.parse_args()

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_path = Path(args.dataset_path)
    minority_numbers = {int(number) for number in args.minority_numbers}
    client = MnistAPFLClient(data_path, minority_numbers, DEVICE)
    fl.client.start_numpy_client(server_address="0.0.0.0:8080", client=client)
