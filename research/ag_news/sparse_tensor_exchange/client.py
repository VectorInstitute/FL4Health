import argparse
import os
from logging import INFO
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import flwr as fl
import torch
import torch.nn as nn
from flwr.common.logger import log
from flwr.common.typing import Config
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification

from fl4health.checkpointing.checkpointer import BestLossTorchCheckpointer
from fl4health.checkpointing.client_module import ClientCheckpointModule
from fl4health.clients.basic_client import TorchInputType
from fl4health.clients.partial_weight_exchange_client import PartialWeightExchangeClient
from fl4health.parameter_exchange.parameter_exchanger_base import ParameterExchanger
from fl4health.parameter_exchange.parameter_selection_criteria import largest_final_magnitude_scores
from fl4health.parameter_exchange.sparse_coo_parameter_exchanger import SparseCooParameterExchanger
from fl4health.reporting.base_reporter import BaseReporter
from fl4health.utils.config import narrow_dict_type
from fl4health.utils.losses import LossMeterType
from fl4health.utils.metrics import Accuracy, Metric
from research.ag_news.client_data import construct_dataloaders


class BertSparseTensorExchangeClient(PartialWeightExchangeClient):
    def __init__(
        self,
        data_path: Path,
        metrics: Sequence[Metric],
        device: torch.device,
        learning_rate: float,
        sparsity_level: float,
        loss_meter_type: LossMeterType = LossMeterType.AVERAGE,
        checkpointer: Optional[ClientCheckpointModule] = None,
        reporters: Sequence[BaseReporter] | None = None,
        store_initial_model: bool = True,
    ) -> None:
        super().__init__(
            data_path=data_path,
            metrics=metrics,
            device=device,
            loss_meter_type=loss_meter_type,
            checkpointer=checkpointer,
            reporters=reporters,
            store_initial_model=store_initial_model,
        )
        self.sparsity_level = sparsity_level
        self.learning_rate: float = learning_rate

    def get_model(self, config: Config) -> nn.Module:
        num_classes = narrow_dict_type(config, "num_classes", int)
        model = BertForSequenceClassification.from_pretrained("google-bert/bert-base-cased", num_labels=num_classes)
        return model.to(self.device)

    def get_data_loaders(self, config: Config) -> Tuple[DataLoader, DataLoader]:
        batch_size = narrow_dict_type(config, "batch_size", int)
        sample_percentage = narrow_dict_type(config, "sample_percentage", float)
        beta = narrow_dict_type(config, "beta", float)
        train_loader, val_loader = construct_dataloaders(batch_size, sample_percentage, beta)
        return train_loader, val_loader

    def get_criterion(self, config: Config) -> _Loss:
        return torch.nn.CrossEntropyLoss()

    def get_optimizer(self, config: Config) -> Optimizer:
        return torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=0.001)

    def get_parameter_exchanger(self, config: Config) -> ParameterExchanger:
        # A different score_gen_function may be passed in to allow for alternative
        # selection criterion.
        parameter_exchanger = SparseCooParameterExchanger(
            sparsity_level=self.sparsity_level,
            score_gen_function=largest_final_magnitude_scores,
        )
        return parameter_exchanger

    def predict(self, input: TorchInputType) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        outputs, features = super().predict(input)
        preds = {}
        preds["prediction"] = outputs["logits"]
        return preds, features


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FL Client Main")
    parser.add_argument(
        "--artifact_dir",
        action="store",
        type=str,
        help="Path to save client artifacts such as logs and model checkpoints",
        required=True,
    )
    parser.add_argument(
        "--dataset_dir",
        action="store",
        type=str,
        help="Path to the local dataset",
        required=True,
    )
    parser.add_argument(
        "--run_name",
        action="store",
        help="Name of the run, model checkpoints will be saved under a subfolder with this name",
        required=True,
    )
    parser.add_argument(
        "--server_address",
        action="store",
        type=str,
        help="Server Address for the clients to communicate with the server through",
        default="0.0.0.0:8080",
    )
    parser.add_argument(
        "--client_number",
        action="store",
        type=int,
        help="Number of the client. Used for checkpointing",
        required=True,
    )
    parser.add_argument(
        "--learning_rate",
        action="store",
        type=float,
        help="Learning rate used by the client",
    )
    parser.add_argument(
        "--sparsity_level",
        action="store",
        type=float,
        help="Sparsity level used for parameter exchange",
    )
    args = parser.parse_args()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_path = Path(args.dataset_dir)

    log(INFO, f"Device to be used: {DEVICE}")
    log(INFO, f"Server Address: {args.server_address}")
    log(INFO, f"Learning Rate: {args.learning_rate}")
    log(INFO, f"Sparsity Level: {args.sparsity_level}")

    # Checkpointing
    checkpoint_dir = os.path.join(args.artifact_dir, args.run_name)
    checkpoint_name = f"client_{args.client_number}_best_model.pkl"
    checkpointer = ClientCheckpointModule(post_aggregation=BestLossTorchCheckpointer(checkpoint_dir, checkpoint_name))

    client = BertSparseTensorExchangeClient(
        data_path,
        [Accuracy("accuracy")],
        DEVICE,
        learning_rate=args.learning_rate,
        sparsity_level=args.sparsity_level,
        checkpointer=checkpointer,
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
