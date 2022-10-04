import argparse
from collections import OrderedDict
from logging import INFO
from pathlib import Path
from typing import Dict, Tuple, Union

import flwr as fl
import torch
from flwr.common.logger import log
from flwr.common.typing import Config, NDArrays, Scalar
from torch.utils.data import DataLoader

from src.examples.fedopt_example.client_data import LabelEncoder, Vocabulary, construct_dataloaders
from src.examples.fedopt_example.metrics import ClientMetrics
from src.examples.fedopt_example.model import LSTM


def train(
    model: LSTM,
    train_loader: DataLoader,
    epochs: int,
    label_encoder: LabelEncoder,
    weight_matrix: torch.Tensor,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Union[float, str]]:
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss(weight=weight_matrix)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.001)
    for epoch in range(epochs):
        running_loss = 0.0
        n_batches = len(train_loader)

        assert train_loader.batch_size is not None
        h0, c0 = model.init_hidden(train_loader.batch_size)
        h0 = h0.to(device)
        c0 = c0.to(device)

        epoch_metrics = ClientMetrics(label_encoder)

        for batch_index, (data, labels) in enumerate(train_loader):
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(data, (h0, c0))
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(out.data, 1)

            # Report some batch loss statistics every so often to track decrease
            if batch_index % 20 == 0:
                log(INFO, f"Batch Index {batch_index} of {n_batches}, Batch loss: {loss.item()}")
            epoch_metrics.update_performance(predicted, labels)

        log_str = epoch_metrics.summarize()
        # Local client logging of epoch results.
        log(
            INFO,
            f"Epoch: {epoch}, Client Training Loss: {running_loss/n_batches}\nClient Training Metrics:{log_str}",
        )
    return epoch_metrics.results


def validate(
    model: LSTM,
    validation_loader: DataLoader,
    label_encoder: LabelEncoder,
    device: torch.device = torch.device("cpu"),
) -> Tuple[float, Dict[str, Union[float, str]]]:
    """Validate the network on the entire validation set."""
    criterion = torch.nn.CrossEntropyLoss()
    loss = 0.0

    assert validation_loader.batch_size is not None
    h0, c0 = model.init_hidden(validation_loader.batch_size)
    h0 = h0.to(device)
    c0 = c0.to(device)

    epoch_metrics = ClientMetrics(label_encoder)

    model.eval()
    with torch.no_grad():
        n_batches = len(validation_loader)
        for data, labels in validation_loader:
            data, labels = data.to(device), labels.to(device)
            out = model(data, (h0, c0))
            loss += criterion(out, labels).item()
            _, predicted = torch.max(out.data, 1)
            epoch_metrics.update_performance(predicted, labels)

    log_str = epoch_metrics.summarize()
    # Local client logging.
    log(
        INFO,
        f"Client Validation Loss: {loss/n_batches}\nClient Validation Metrics:{log_str}",
    )
    return loss / n_batches, epoch_metrics.results


class NewsClassifier(fl.client.NumPyClient):
    def __init__(
        self,
        data_path: Path,
        device: torch.device,
    ) -> None:
        self.data_path = data_path
        self.device = device
        self.intialized = False

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

    def setup_client(self, config: Config) -> None:
        sequence_length = config["sequence_length"]
        batch_size = config["batch_size"]
        vocab_dimension = config["vocab_dimension"]
        hidden_size = config["hidden_size"]
        vocabulary = Vocabulary.from_json(config["vocabulary"])
        label_encoder = LabelEncoder.from_json(config["label_encoder"])

        train_loader, validation_loader, num_examples, weight_matrix = construct_dataloaders(
            self.data_path, vocabulary, label_encoder, sequence_length, batch_size
        )

        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.num_examples = num_examples
        self.label_encoder = label_encoder
        self.weight_matrix = weight_matrix

        self.setup_model(vocabulary.vocabulary_size, vocab_dimension, hidden_size)

    def setup_model(self, vocab_size: int, vocab_dimension: int, hidden_size: int) -> None:
        self.model = LSTM(vocab_size, vocab_dimension, hidden_size)

    def fit(self, parameters: NDArrays, config: Config) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        if not self.intialized:
            self.setup_client(config)
        # Once the model is created model weights are initialized from server
        self.set_parameters(parameters, config)

        fit_metrics = train(
            self.model,
            self.train_loader,
            config["local_epochs"],
            self.label_encoder,
            self.weight_matrix,
            device=self.device,
        )
        # Result should contain local parameters, number of examples on client, and a dictionary holding metrics
        # calculation results.
        return (
            self.get_parameters(config),
            self.num_examples["train_set"],
            fit_metrics,
        )

    def evaluate(self, parameters: NDArrays, config: Config) -> Tuple[float, int, Dict[str, Scalar]]:
        self.set_parameters(parameters, config)
        loss, evaluate_metrics = validate(self.model, self.validation_loader, self.label_encoder, self.device)
        # Result should return the loss, number of examples on client, and a dictionary holding metrics
        # calculation results.
        return (
            loss,
            self.num_examples["validation_set"],
            evaluate_metrics,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FL Client Main")
    parser.add_argument("--dataset_path", action="store", type=str, help="Path to the local dataset")
    args = parser.parse_args()

    # Load model and data
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_path = Path(args.dataset_path)
    client = NewsClassifier(data_path, DEVICE)
    fl.client.start_numpy_client(server_address="0.0.0.0:8080", client=client)
