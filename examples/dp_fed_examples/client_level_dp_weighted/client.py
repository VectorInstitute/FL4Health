import argparse
from collections import OrderedDict
from logging import INFO
from pathlib import Path
from typing import Dict, Tuple

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
from flwr.common.logger import log
from flwr.common.typing import Config, NDArrays, Scalar
from torch.utils.data import DataLoader

from examples.dp_fed_examples.client_level_dp_weighted.data import load_data
from examples.models.logistic_regression import LogisticRegression
from fl4health.clients.clipping_client import NumpyClippingClient


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
    def __init__(
        self,
        data_path: Path,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.device = device
        self.data_path = data_path
        self.initialized = False
        self.train_loader: DataLoader

    def get_parameters(self, config: Config) -> NDArrays:
        # Determines which weights are sent back to the server for aggregation.
        # Currently sending all of them ordered by state_dict keys
        # NOTE: Order matters, because it is relied upon by set_parameters below
        model_weights = [val.cpu().numpy() for _, val in self.model.state_dict().items()]
        # Clipped the weights and store clipping information in parameters
        clipped_weight_update, clipping_bit = self.compute_weight_update_and_clip(model_weights)
        return clipped_weight_update + [np.array([clipping_bit])]

    def set_parameters(self, parameters: NDArrays, config: Config) -> None:
        # Sets the local model parameters transfered from the server. The state_dict is
        # reconstituted because parameters is simply a list of bytes
        # The last entry in the parameters list is assumed to be a clipping bound (even if we're evaluating)
        server_model_parameters = parameters[:-1]
        params_dict = zip(self.model.state_dict().keys(), server_model_parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

        # Store the starting parameters without clipping bound before client optimization steps
        self.current_weights = server_model_parameters

        # Expectation is that the last entry in the parameters NDArrays is a clipping bound
        clipping_bound = parameters[-1]
        self.clipping_bound = float(clipping_bound)

    def setup_client(self, config: Config) -> None:
        self.batch_size = config["batch_size"]
        self.local_epochs = config["local_epochs"]
        self.adaptive_clipping = config["adaptive_clipping"]
        self.scaler_bytes = config["scaler"]

        train_loader, validation_loader, num_examples = load_data(self.data_path, self.batch_size, self.scaler_bytes)

        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.num_examples = num_examples
        self.model = LogisticRegression(input_dim=31, output_dim=1).to(self.device)
        self.initialized = True

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
    client = HospitalClient(data_path, DEVICE)
    fl.client.start_numpy_client(server_address="0.0.0.0:8080", client=client)
