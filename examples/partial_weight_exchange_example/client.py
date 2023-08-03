import argparse
import copy
from logging import INFO
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import flwr as fl
import torch
import torch.nn as nn
from flwr.common.logger import log
from flwr.common.typing import Config, NDArrays, Scalar
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader
from torcheval.metrics.functional import multiclass_f1_score
from torchtext.models import ROBERTA_BASE_ENCODER, RobertaClassificationHead

from examples.partial_weight_exchange_example.client_data import construct_dataloaders
from fl4health.clients.basic_client import BasicClient
from fl4health.parameter_exchange.layer_exchanger import NormDriftParameterExchanger
from fl4health.utils.metrics import Accuracy, AverageMeter, Metric


class TransformerPartialExchangeClient(BasicClient):
    def __init__(
        self,
        data_path: Path,
        metrics: Sequence[Metric],
        device: torch.device,
    ) -> None:
        super().__init__(data_path, metrics, device)
        self.model: nn.Module
        self.initial_model: nn.Module
        self.train_loader: DataLoader
        self.val_loader: DataLoader
        self.test_loader: DataLoader
        self.criterion: _Loss
        self.optimizer: torch.optim.Optimizer
        self.parameter_exchanger: NormDriftParameterExchanger

    def evaluate(self, parameters: NDArrays, config: Config) -> Tuple[float, int, Dict[str, Scalar]]:
        if not self.initialized:
            self.setup_client(config)

        self.set_parameters(parameters, config)
        testing = self.narrow_config_type(config, "testing", bool)
        num_classes = num_classes = self.narrow_config_type(config, "num_classes", int)

        if not testing:
            meter = AverageMeter(self.metrics, "val_meter")
            loss, metric_values = self.validate(meter)
            # EvaluateRes should return the loss, number of examples on client, and a dictionary holding metrics
            # calculation results.
            return (
                loss,
                self.num_examples["validation_set"],
                metric_values,
            )
        else:
            loss, accuracy, f1_scores = self.test(num_classes=num_classes)
            test_res_dict: Dict[str, Scalar] = {f"class {c} f1_score": f1_scores[c] for c in range(len(f1_scores))}
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
        norm_threshold = self.narrow_config_type(config, "norm_threshold", float)
        exchange_percentage = self.narrow_config_type(config, "exchange_percentage", float)
        sample_percentage = self.narrow_config_type(config, "sample_percentage", float)
        beta = self.narrow_config_type(config, "beta", float)

        self.parameter_exchanger = NormDriftParameterExchanger(
            norm_threshold, exchange_percentage, normalize, filter_by_percentage
        )

        self.setup_model(num_classes)

        train_loader, val_loader, test_loader, num_examples = construct_dataloaders(
            self.data_path, batch_size, sequence_length, sample_percentage, beta
        )

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.num_examples = num_examples
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.00001, weight_decay=0.001)

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
        # stores the values of the model parameters at the beginning of each training round.
        self._align_model_parameters(self.initial_model, self.model)

    def _align_model_parameters(self, initial_model: nn.Module, target_model: nn.Module) -> None:
        target_model_states = target_model.state_dict()
        initial_model_states = initial_model.state_dict()
        for param_name, param in target_model_states.items():
            initial_model_states[param_name] = param
        initial_model.load_state_dict(initial_model_states, strict=True)

    def test(self, num_classes: int) -> Tuple[float, float, List[float]]:
        self.model.eval()
        with torch.no_grad():
            n_total = 0
            n_correct = 0
            n_batches = 0
            total_loss = 0.0

            preds_lst = []
            targets_lst = []

            for inputs, targets in self.test_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(inputs)
                preds = torch.argmax(outputs, dim=1)

                loss = self.criterion(outputs, targets)

                preds_lst.append(preds)
                targets_lst.append(targets)

                total_loss += loss.item()
                n_total += targets.size(0)
                n_correct += int((preds == targets).sum().item())
                n_batches += 1

            test_loss = total_loss / n_batches
            accuracy = n_correct / n_total

            f1_score = multiclass_f1_score(
                torch.cat(preds_lst), torch.cat(targets_lst), num_classes=num_classes, average=None
            )

            log(
                INFO,
                f"Client Test Loss: {test_loss},"
                f"Client Test Accuracy: {accuracy}, Client Test f1 score: {f1_score.tolist()}",
            )

            return test_loss, accuracy, f1_score.tolist()


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

    client = TransformerPartialExchangeClient(data_path, [Accuracy("accuracy")], DEVICE)
    # grpc_max_message_length is reset here so the entire model can be exchanged between the server and clients.
    # Note that the server must be started with the same grpc_max_message_length. Otherwise communication
    # of larger messages would still be blocked.
    fl.client.start_numpy_client(server_address=args.server_address, client=client, grpc_max_message_length=1600000000)
