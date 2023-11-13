import argparse
import os
from logging import INFO
from pathlib import Path
from typing import Optional, Sequence, Tuple

import flwr as fl
import torch
import torch.nn as nn
from flamby.datasets.fed_ixi import BATCH_SIZE, LR, NUM_CLIENTS, BaselineLoss
from flwr.common.logger import log
from flwr.common.typing import Config
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from fl4health.checkpointing.checkpointer import (
    BestMetricTorchCheckpointer,
    LatestTorchCheckpointer,
    TorchCheckpointer,
)
from fl4health.clients.fenda_client import FendaClient
from fl4health.utils.losses import LossMeterType
from fl4health.utils.metrics import BinarySoftDiceCoefficient, Metric, MetricMeterType
from research.flamby.fed_ixi.fenda.fenda_model import FedIxiFendaModel
from research.flamby.flamby_data_utils import construct_fed_ixi_train_val_datasets


class FedIxiFendaClient(FendaClient):
    def __init__(
        self,
        data_path: Path,
        metrics: Sequence[Metric],
        device: torch.device,
        client_number: int,
        learning_rate: float,
        loss_meter_type: LossMeterType = LossMeterType.AVERAGE,
        metric_meter_type: MetricMeterType = MetricMeterType.ACCUMULATION,
        checkpointer: Optional[TorchCheckpointer] = None,
        cos_sim_activate: bool = False,
        contrastive_activate: bool = False,
        perfcl_activate: bool = False,
    ) -> None:
        super().__init__(
            data_path=data_path,
            metrics=metrics,
            device=device,
            loss_meter_type=loss_meter_type,
            metric_meter_type=metric_meter_type,
            checkpointer=checkpointer,
        )
        self.client_number = client_number
        self.learning_rate = learning_rate
        if cos_sim_activate:
            self.cos_sim_loss_weight = 100.0
        if contrastive_activate:
            self.contrastive_loss_weight = 10.0
        if perfcl_activate:
            self.perfcl_loss_weights = (10.0, 10.0)

        assert 0 <= client_number < NUM_CLIENTS
        log(INFO, f"Client Name: {self.client_name}, Client Number: {self.client_number}")

    def get_data_loaders(self, config: Config) -> Tuple[DataLoader, DataLoader]:
        train_dataset, validation_dataset = construct_fed_ixi_train_val_datasets(
            self.client_number, str(self.data_path)
        )
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False)
        return train_loader, val_loader

    def get_model(self, config: Config) -> nn.Module:
        model: nn.Module = FedIxiFendaModel().to(self.device)
        return model

    def get_optimizer(self, config: Config) -> Optimizer:
        return torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)

    def get_criterion(self, config: Config) -> _Loss:
        return BaselineLoss()


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
        help="Path to the preprocessed FedIXI Dataset (ex. path/to/fedixi/dataset)",
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
        help="Number of the client for dataset loading (should be 0-2 for FedIXI)",
        required=True,
    )
    parser.add_argument(
        "--learning_rate", action="store", type=float, help="Learning rate for local optimization", default=LR
    )
    parser.add_argument("--cos_sim_loss", action="store_true", help="Activate Cosine Similarity loss")
    parser.add_argument("--contrastive_loss", action="store_true", help="Activate Contrastive loss")
    parser.add_argument("--perfcl_loss", action="store_true", help="Activate PerFCL loss")
    parser.add_argument(
        "--no_federated_checkpointing",
        action="store_true",
        help="boolean to indicate whether we're evaluating an APFL model or not, as those model have special args",
    )
    args = parser.parse_args()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(INFO, f"Device to be used: {DEVICE}")
    log(INFO, f"Server Address: {args.server_address}")
    log(INFO, f"Learning Rate: {args.learning_rate}")
    log(INFO, f"Performing Federated Checkpointing: {not args.no_federated_checkpointing}")

    federated_checkpointing = not args.no_federated_checkpointing
    checkpoint_dir = os.path.join(args.artifact_dir, args.run_name)
    checkpoint_name = f"client_{args.client_number}_best_model.pkl"
    checkpointer = (
        BestMetricTorchCheckpointer(checkpoint_dir, checkpoint_name, maximize=False)
        if federated_checkpointing
        else LatestTorchCheckpointer(checkpoint_dir, checkpoint_name)
    )

    client = FedIxiFendaClient(
        data_path=Path(args.dataset_dir),
        metrics=[BinarySoftDiceCoefficient("FedIXI_dice")],
        device=DEVICE,
        client_number=args.client_number,
        learning_rate=args.learning_rate,
        checkpointer=checkpointer,
        cos_sim_activate=args.cos_sim_loss,
        contrastive_activate=args.contrastive_loss,
        perfcl_activate=args.perfcl_loss,
    )

    fl.client.start_numpy_client(server_address=args.server_address, client=client)

    # Shutdown the client gracefully
    client.shutdown()
