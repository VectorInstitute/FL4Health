import argparse
from logging import INFO
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import flwr as fl
import torch
import torch.nn as nn
from flwr.common.logger import log
from flwr.common.typing import Config, NDArrays, Scalar
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torcheval.metrics.functional import multiclass_f1_score
from torchtext.models import ROBERTA_BASE_ENCODER, RobertaClassificationHead

from examples.partial_weight_exchange_example.client_data import construct_dataloaders
from fl4health.checkpointing.checkpointer import TorchCheckpointer
from fl4health.clients.dynamic_weight_exchange_client import DynamicWeightExchangeClient
from fl4health.utils.losses import LossMeterType
from fl4health.utils.metrics import Accuracy, Metric


class TransformerPartialExchangeClient(DynamicWeightExchangeClient):
    def __init__(
        self,
        data_path: Path,
        metrics: Sequence[Metric],
        device: torch.device,
        loss_meter_type: LossMeterType = LossMeterType.AVERAGE,
        checkpointer: Optional[TorchCheckpointer] = None,
        adaptive_exchange_percentage: bool = False,
        exchange_percentage_delta: float = 0.05,
    ) -> None:
        super().__init__(
            data_path=data_path,
            metrics=metrics,
            device=device,
            loss_meter_type=loss_meter_type,
            checkpointer=checkpointer,
            adaptive_exchange_percentage=adaptive_exchange_percentage,
            exchange_percentage_delta=exchange_percentage_delta,
        )
        self.test_loader: DataLoader

    def evaluate(self, parameters: NDArrays, config: Config) -> Tuple[float, int, Dict[str, Scalar]]:
        if not self.initialized:
            self.setup_client(config)

        self.set_parameters(parameters, config)
        testing = self.narrow_config_type(config, "testing", bool)
        num_classes = num_classes = self.narrow_config_type(config, "num_classes", int)

        if not testing:
            return super().evaluate(parameters, config)
        else:
            loss, accuracy, f1_scores = self.test(num_classes=num_classes)
            test_res_dict: Dict[str, Scalar] = {f"class {c} f1_score": f1_scores[c] for c in range(len(f1_scores))}
            test_res_dict["accuracy"] = accuracy
            return (loss, self.num_examples["test_set"], test_res_dict)

    def get_model(self, config: Config) -> nn.Module:
        num_classes = self.narrow_config_type(config, "num_classes", int)
        classifier_head = RobertaClassificationHead(num_classes=num_classes, input_dim=768)
        model = ROBERTA_BASE_ENCODER.get_model(head=classifier_head).to(self.device)
        return model

    def get_data_loaders(self, config: Config) -> Tuple[DataLoader, DataLoader]:
        batch_size = self.narrow_config_type(config, "batch_size", int)
        sequence_length = self.narrow_config_type(config, "sequence_length", int)
        sample_percentage = self.narrow_config_type(config, "sample_percentage", float)
        beta = self.narrow_config_type(config, "beta", float)
        train_loader, val_loader, test_loader, num_examples = construct_dataloaders(
            self.data_path, batch_size, sequence_length, sample_percentage, beta
        )
        self.test_loader = test_loader
        self.num_examples = num_examples
        return train_loader, val_loader

    def get_criterion(self, config: Config) -> _Loss:
        return torch.nn.CrossEntropyLoss()

    def get_optimizer(self, config: Config) -> Optimizer:
        return torch.optim.AdamW(self.model.parameters(), lr=0.00001, weight_decay=0.001)

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

    client = TransformerPartialExchangeClient(
        data_path, [Accuracy("accuracy")], DEVICE, adaptive_exchange_percentage=True
    )
    # grpc_max_message_length is reset here so the entire model can be exchanged between the server and clients.
    # Note that the server must be started with the same grpc_max_message_length. Otherwise communication
    # of larger messages would still be blocked.
    fl.client.start_numpy_client(server_address=args.server_address, client=client, grpc_max_message_length=1600000000)

    client.shutdown()
