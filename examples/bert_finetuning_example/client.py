import argparse
from logging import INFO
from pathlib import Path

import flwr as fl
import torch
from flwr.common.logger import log
from flwr.common.typing import Config
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification

from examples.bert_finetuning_example.client_data import construct_dataloaders
from fl4health.clients.basic_client import BasicClient, TorchInputType
from fl4health.metrics import Accuracy
from fl4health.utils.config import narrow_dict_type


class BertClient(BasicClient):
    def __init__(
        self,
        metrics: list,
        device: torch.device,
        learning_rate: float,
    ) -> None:
        super().__init__(
            data_path=Path("null"),  # Path is ignored here, as we use HF load_dataset
            metrics=metrics,
            device=device,
            progress_bar=True,
        )
        self.learning_rate: float = learning_rate

    def get_model(self, config: Config) -> nn.Module:
        num_classes = narrow_dict_type(config, "num_classes", int)
        model = BertForSequenceClassification.from_pretrained("google-bert/bert-base-cased", num_labels=num_classes)
        return model.to(self.device)

    def get_data_loaders(self, config: Config) -> tuple[DataLoader, DataLoader]:
        batch_size = narrow_dict_type(config, "batch_size", int)
        sample_percentage = narrow_dict_type(config, "sample_percentage", float)
        beta = narrow_dict_type(config, "beta", float)
        train_loader, val_loader = construct_dataloaders(batch_size, sample_percentage, beta)
        return train_loader, val_loader

    def get_criterion(self, config: Config) -> _Loss:
        return torch.nn.CrossEntropyLoss()

    def get_optimizer(self, config: Config) -> Optimizer:
        return torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=0.001)

    def predict(self, input: TorchInputType) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        # Here the predict method is overwritten in order
        # to rename the key to match what comes with the hugging face datasets.
        outputs, features = super().predict(input)
        preds = {"prediction": outputs["logits"]}
        return preds, features


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FL Client Main")
    parser.add_argument(
        "--server_address",
        action="store",
        type=str,
        help="Server Address for the clients to communicate with the server through",
        default="0.0.0.0:8080",
    )
    parser.add_argument(
        "--artifact_dir",
        action="store",
        type=str,
        help="Path to save client artifacts such as logs and model checkpoints",
        default="examples/bert_finetuning_example",
    )
    parser.add_argument(
        "--learning_rate",
        action="store",
        type=float,
        help="Learning rate used by the client",
        default=0.0001,
    )
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    log(INFO, f"Device to be used: {device}")
    log(INFO, f"Server Address: {args.server_address}")
    log(INFO, f"Learning Rate: {args.learning_rate}")
    log(INFO, f"Artifact Directory: {args.artifact_dir}")

    client = BertClient(
        [Accuracy("accuracy")],
        device,
        learning_rate=args.learning_rate,
    )
    # grpc_max_message_length is reset here so the entire model can be exchanged between the server and clients.
    # Note that the server must be started with the same grpc_max_message_length. Otherwise communication
    # of larger messages would still be blocked.
    fl.client.start_client(
        server_address=args.server_address,
        client=client.to_client(),
        grpc_max_message_length=1600000000,
    )
    client.shutdown()
