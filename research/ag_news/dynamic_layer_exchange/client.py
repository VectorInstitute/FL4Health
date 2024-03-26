import argparse
import os
from logging import INFO
from pathlib import Path
from typing import Optional, Sequence, Tuple

import flwr as fl
import torch
import torch.nn as nn
from flwr.common.logger import log
from flwr.common.typing import Config
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification

from fl4health.checkpointing.checkpointer import BestMetricTorchCheckpointer, TorchCheckpointer
from fl4health.clients.partial_weight_exchange_client import PartialWeightExchangeClient
from fl4health.parameter_exchange.layer_exchanger import DynamicLayerExchanger
from fl4health.parameter_exchange.parameter_exchanger_base import ParameterExchanger
from fl4health.parameter_exchange.parameter_selection_criteria import LayerSelectionFunctionConstructor
from fl4health.reporting.metrics import MetricsReporter
from fl4health.utils.losses import LossMeterType
from fl4health.utils.metrics import Accuracy, Metric
from research.ag_news.dynamic_layer_exchange.client_data import construct_dataloaders


class BertDynamicLayerExchangeClient(PartialWeightExchangeClient):
    def __init__(
        self,
        data_path: Path,
        metrics: Sequence[Metric],
        device: torch.device,
        learning_rate: float,
        exchange_percentage: float,
        loss_meter_type: LossMeterType = LossMeterType.AVERAGE,
        checkpointer: Optional[TorchCheckpointer] = None,
        metrics_reporter: Optional[MetricsReporter] = None,
        store_initial_model: bool = True,
    ) -> None:
        super().__init__(
            data_path=data_path,
            metrics=metrics,
            device=device,
            loss_meter_type=loss_meter_type,
            checkpointer=checkpointer,
            metrics_reporter=metrics_reporter,
            store_initial_model=store_initial_model,
        )
        assert 0 < exchange_percentage <= 1.0
        self.exchange_percentage = exchange_percentage
        self.learning_rate: float = learning_rate

    def get_model(self, config: Config) -> nn.Module:
        num_classes = self.narrow_config_type(config, "num_classes", int)
        model = BertForSequenceClassification.from_pretrained("google-bert/bert-base-cased", num_labels=num_classes)
        return model.to(self.device)

    def get_data_loaders(self, config: Config) -> Tuple[DataLoader, DataLoader]:
        batch_size = self.narrow_config_type(config, "batch_size", int)
        sample_percentage = self.narrow_config_type(config, "sample_percentage", float)
        beta = self.narrow_config_type(config, "beta", float)
        train_loader, val_loader = construct_dataloaders(batch_size, sample_percentage, beta)
        return train_loader, val_loader

    def get_criterion(self, config: Config) -> _Loss:
        return torch.nn.CrossEntropyLoss()

    def get_optimizer(self, config: Config) -> Optimizer:
        return torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=0.001)

    def get_parameter_exchanger(self, config: Config) -> ParameterExchanger:
        """
        This method configures and instantiates a NormDriftParameterExchanger to be used in dynamic weight exchange.

        Args:
            config (Config): Configuration used to setup the weight exchanger properties for dynamic exchange

        Returns:
            ParameterExchanger: This exchanger handles the exchange orchestration between clients and server during
                federated training
        """
        normalize = self.narrow_config_type(config, "normalize", bool)
        filter_by_percentage = self.narrow_config_type(config, "filter_by_percentage", bool)
        norm_threshold = self.narrow_config_type(config, "norm_threshold", float)
        select_drift_more = self.narrow_config_type(config, "select_drift_more", bool)
        selection_function_constructor = LayerSelectionFunctionConstructor(
            norm_threshold=norm_threshold,
            exchange_percentage=self.exchange_percentage,
            normalize=normalize,
            select_drift_more=select_drift_more,
        )
        layer_selection_function = (
            selection_function_constructor.select_by_percentage()
            if filter_by_percentage
            else selection_function_constructor.select_by_threshold()
        )
        parameter_exchanger = DynamicLayerExchanger(layer_selection_function=layer_selection_function)
        return parameter_exchanger


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
        "--exchange_percentage",
        action="store",
        type=float,
        help="Percentage of the number of tensors that are exchanged with the server",
    )
    args = parser.parse_args()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_path = Path(args.dataset_dir)

    log(INFO, f"Device to be used: {DEVICE}")
    log(INFO, f"Server Address: {args.server_address}")
    log(INFO, f"Learning Rate: {args.learning_rate}")
    log(INFO, f"Exchange Percentage: {args.exchange_percentage}")

    # Checkpointing
    checkpoint_dir = os.path.join(args.artifact_dir, args.run_name)
    checkpoint_name = f"client_{args.client_number}_best_model.pkl"
    checkpointer = BestMetricTorchCheckpointer(checkpoint_dir, checkpoint_name, maximize=False)

    client = BertDynamicLayerExchangeClient(
        data_path,
        [Accuracy("accuracy")],
        DEVICE,
        learning_rate=args.learning_rate,
        exchange_percentage=args.exchange_percentage,
        checkpointer=checkpointer,
    )
    # grpc_max_message_length is reset here so the entire model can be exchanged between the server and clients.
    # Note that the server must be started with the same grpc_max_message_length. Otherwise communication
    # of larger messages would still be blocked.
    fl.client.start_numpy_client(server_address=args.server_address, client=client, grpc_max_message_length=1600000000)

    client.shutdown()
