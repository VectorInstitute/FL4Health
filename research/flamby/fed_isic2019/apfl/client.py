import argparse
import os
from logging import INFO
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import flwr as fl
import torch
import torch.nn as nn
from flamby.datasets.fed_isic2019 import BATCH_SIZE, LR, NUM_CLIENTS, BaselineLoss
from flwr.common.logger import log
from flwr.common.typing import Config
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from fl4health.checkpointing.checkpointer import BestMetricTorchCheckpointer, TorchCheckpointer
from fl4health.clients.apfl_client import ApflClient
from fl4health.model_bases.apfl_base import APFLModule
from fl4health.utils.losses import LossMeterType
from fl4health.utils.metrics import BalancedAccuracy, Metric, MetricMeterType
from research.flamby.fed_isic2019.apfl.apfl_model import APFLEfficientNet
from research.flamby.flamby_data_utils import construct_fedisic_train_val_datasets


class FedIsic2019ApflClient(ApflClient):
    def __init__(
        self,
        data_path: Path,
        metrics: Sequence[Metric],
        device: torch.device,
        client_number: int,
        learning_rate: float,
        alpha_learning_rate: float,
        loss_meter_type: LossMeterType = LossMeterType.AVERAGE,
        metric_meter_type: MetricMeterType = MetricMeterType.ACCUMULATION,
        checkpointer: Optional[TorchCheckpointer] = None,
    ) -> None:
        super().__init__(
            data_path=data_path,
            metrics=metrics,
            device=device,
            loss_meter_type=loss_meter_type,
            metric_meter_type=metric_meter_type,
            checkpointer=checkpointer,
        )
        assert 0 <= client_number < NUM_CLIENTS

        self.learning_rate = learning_rate
        self.alpha_learning_rate = alpha_learning_rate
        self.client_number = client_number

    def get_data_loaders(self, config: Config) -> Tuple[DataLoader, DataLoader]:
        train_dataset, validation_dataset = construct_fedisic_train_val_datasets(
            self.client_number, str(self.data_path)
        )
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False)
        return train_loader, val_loader

    def get_criterion(self, config: Config) -> _Loss:
        return BaselineLoss()

    def get_model(self, config: Config) -> nn.Module:
        model: APFLModule = APFLModule(
            APFLEfficientNet(frozen_blocks=13, turn_off_bn_tracking=False), alpha_lr=self.alpha_learning_rate
        ).to(self.device)
        return model

    def get_optimizer(self, config: Config) -> Dict[str, Optimizer]:
        local_optimizer: Optimizer = torch.optim.AdamW(self.model.local_model.parameters(), lr=self.learning_rate)
        global_optimizer: Optimizer = torch.optim.AdamW(self.model.global_model.parameters(), lr=self.learning_rate)
        return {"local": local_optimizer, "global": global_optimizer}


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
        help="Path to the preprocessed FedIsic2019 Dataset (ex. path/to/fedisic2019)",
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
        help="Number of the client for dataset loading (should be 0-5 for FedIsic2019)",
        required=True,
    )
    parser.add_argument(
        "--learning_rate", action="store", type=float, help="Learning rate for local optimization", default=LR
    )
    parser.add_argument(
        "--alpha_learning_rate", action="store", type=float, help="Learning rate for the APFL alpha", default=0.01
    )
    args = parser.parse_args()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(INFO, f"Device to be used: {DEVICE}")
    log(INFO, f"Server Address: {args.server_address}")
    log(INFO, f"Learning Rate: {args.learning_rate}")
    log(INFO, f"Alpha Learning Rate: {args.alpha_learning_rate}")

    checkpoint_dir = os.path.join(args.artifact_dir, args.run_name)
    checkpoint_name = f"client_{args.client_number}_best_model.pkl"
    checkpointer = BestMetricTorchCheckpointer(checkpoint_dir, checkpoint_name, maximize=False)

    client = FedIsic2019ApflClient(
        data_path=Path(args.dataset_dir),
        metrics=[BalancedAccuracy("FedIsic2019_balanced_accuracy")],
        device=DEVICE,
        client_number=args.client_number,
        learning_rate=args.learning_rate,
        alpha_learning_rate=args.alpha_learning_rate,
        checkpointer=checkpointer,
    )
    fl.client.start_numpy_client(server_address=args.server_address, client=client)

    # Shutdown the client gracefully
    client.shutdown()
