import argparse
import copy
from logging import INFO
from pathlib import Path
from typing import Dict, Tuple

import flwr as fl
import torch
import torch.nn as nn
from flwr.common.logger import log
from flwr.common.typing import Config, NDArrays, Scalar
from torchtext.models import ROBERTA_BASE_ENCODER, RobertaClassificationHead

from examples.partial_weight_exchange_example.client_data import construct_dataloaders
from examples.partial_weight_exchange_example.trainer import test, train, validate
from fl4health.clients.numpy_fl_client import NumpyFlClient
from fl4health.parameter_exchange.layer_exchanger import NormDriftParameterExchanger


class TransformerPartialExchangeClient(NumpyFlClient):
    def __init__(
        self, data_path: Path, device: torch.device, exchange_percentage: float, norm_threshold: float
    ) -> None:
        super().__init__(data_path, device)
        self.parameter_exchanger: NormDriftParameterExchanger = NormDriftParameterExchanger(
            norm_threshold=norm_threshold, exchange_percentage=exchange_percentage
        )

    def fit(self, parameters: NDArrays, config: Config) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        if not self.initialized:
            self.setup_client(config)
        self.set_parameters(parameters, config)

        local_epochs = self.narrow_config_type(config, "local_epochs", int)
        accuracy = train(
            self.model,
            self.train_loader,
            nn.CrossEntropyLoss(),
            n_epochs=local_epochs,
            device=self.device,
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
        testing = self.narrow_config_type(config, "testing", bool)
        num_classes = num_classes = self.narrow_config_type(config, "num_classes", int)

        if not testing:
            loss, accuracy = validate(self.model, self.validation_loader, nn.CrossEntropyLoss(), device=self.device)
            # EvaluateRes should return the loss, number of examples on client, and a dictionary holding metrics
            # calculation results.
            return (
                loss,
                self.num_examples["validation_set"],
                {"accuracy": accuracy},
            )
        else:
            loss, accuracy, f1_score = test(
                self.model, self.test_loader, nn.CrossEntropyLoss(), device=self.device, num_classes=num_classes
            )

            test_res_dict: Dict[str, Scalar] = {f"class {c} f1_score": f1_score[c] for c in range(len(f1_score))}
            test_res_dict["accuracy"] = accuracy
            return (loss, self.num_examples["test_set"], test_res_dict)

    def setup_model(self, num_classes: int) -> None:
        classifier_head = RobertaClassificationHead(num_classes=num_classes, input_dim=768)
        self.model = ROBERTA_BASE_ENCODER.get_model(head=classifier_head)
        self.model.to(self.device)
        self.initial_model = copy.deepcopy(self.model).to(self.device)

    def setup_client(self, config: Config) -> None:
        super().setup_client(config)
        batch_size = self.narrow_config_type(config, "batch_size", int)
        sequence_length = self.narrow_config_type(config, "sequence_length", int)
        num_classes = self.narrow_config_type(config, "num_classes", int)
        normalize = self.narrow_config_type(config, "normalize", bool)
        filter_by_percentage = self.narrow_config_type(config, "filter_by_percentage", bool)
        sample_percentage = self.narrow_config_type(config, "sample_percentage", float)
        beta = self.narrow_config_type(config, "beta", float)

        self.parameter_exchanger.set_normalization_mode(normalize)
        self.parameter_exchanger.set_filter_mode(filter_by_percentage)

        self.setup_model(num_classes)

        train_loader, val_loader, test_loader, num_examples = construct_dataloaders(
            self.data_path, batch_size, sequence_length, sample_percentage, beta
        )

        self.train_loader = train_loader
        self.validation_loader = val_loader
        self.test_loader = test_loader
        self.num_examples = num_examples

    def get_parameters(self, config: Config) -> NDArrays:
        # Determines which weights are sent back to the server for aggregation. This uses a parameter exchanger to
        # determine parameters sent
        assert self.model is not None and self.parameter_exchanger is not None
        return self.parameter_exchanger.push_parameters(self.model, self.initial_model, config=config)

    def set_parameters(self, parameters: NDArrays, config: Config) -> None:
        # Sets the local model parameters transfered from the server using a parameter exchanger to coordinate how
        # parameters are set
        assert self.model is not None and self.parameter_exchanger is not None
        self.parameter_exchanger.pull_parameters(parameters, self.model, config)
        # this second pull stores the values of the model parameters at the beginning of each training round.
        self.parameter_exchanger.pull_parameters(parameters, self.initial_model, config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FL Client Main")
    parser.add_argument(
        "--dataset_path", action="store", type=str, help="Path to the local dataset", default="examples/datasets"
    )
    parser.add_argument(
        "--server_address",
        action="store",
        type=str,
        help="Server Address for the clients to communicate with the server through",
        default="0.0.0.0:8080",
    )
    parser.add_argument("--exchange_percentage", action="store", type=float, default=0.1)
    parser.add_argument("--norm_threshold", action="store", type=float, default=24.5)
    args = parser.parse_args()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_path = Path(args.dataset_path)

    log(INFO, f"Device to be used: {DEVICE}")
    log(INFO, f"Server Address: {args.server_address}")

    client = TransformerPartialExchangeClient(data_path, DEVICE, args.exchange_percentage, args.norm_threshold)
    # grpc_max_message_length is reset here so the entire model can be exchanged between the server and clients
    fl.client.start_numpy_client(server_address=args.server_address, client=client, grpc_max_message_length=1600000000)
