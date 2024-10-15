import argparse
import os
from collections import OrderedDict
from logging import INFO
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import flwr as fl
import torch
import torch.nn as nn
from flamby.datasets.fed_isic2019 import BATCH_SIZE, LR, NUM_CLIENTS, Baseline, BaselineLoss
from flwr.common.logger import log
from flwr.common.typing import Config
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from fl4health.checkpointing.checkpointer import BestLossTorchCheckpointer
from fl4health.checkpointing.client_module import ClientCheckpointModule
from fl4health.clients.deep_mmd_clients.ditto_deep_mmd_client import DittoDeepMmdClient
from fl4health.utils.losses import LossMeterType
from fl4health.utils.metrics import BalancedAccuracy, Metric
from fl4health.utils.random import set_all_random_seeds
from research.flamby.flamby_data_utils import construct_fedisic_train_val_datasets

FED_ISIC2019_BASELINE_LAYERS: OrderedDict[str, int] = OrderedDict()
for i in range(16):
    FED_ISIC2019_BASELINE_LAYERS[f"base_model._blocks.{i}"] = 64
FED_ISIC2019_BASELINE_LAYERS["base_model._dropout"] = 1280


class FedIsic2019DittoClient(DittoDeepMmdClient):
    def __init__(
        self,
        data_path: Path,
        metrics: Sequence[Metric],
        device: torch.device,
        client_number: int,
        learning_rate: float,
        loss_meter_type: LossMeterType = LossMeterType.AVERAGE,
        deep_mmd_loss_weight: float = 10,
        deep_mmd_loss_depth: int = 1,
        checkpointer: Optional[ClientCheckpointModule] = None,
    ) -> None:
        size_feature_extraction_layers = OrderedDict(
            list(FED_ISIC2019_BASELINE_LAYERS.items())[-1 * deep_mmd_loss_depth :]
        )
        super().__init__(
            data_path=data_path,
            metrics=metrics,
            device=device,
            loss_meter_type=loss_meter_type,
            checkpointer=checkpointer,
            deep_mmd_loss_weight=deep_mmd_loss_weight,
            size_feature_extraction_layers=size_feature_extraction_layers,
        )
        self.client_number = client_number
        self.learning_rate: float = learning_rate

        assert 0 <= client_number < NUM_CLIENTS
        log(INFO, f"Client Name: {self.client_name}, Client Number: {self.client_number}")

    def get_data_loaders(self, config: Config) -> Tuple[DataLoader, DataLoader]:
        train_dataset, validation_dataset = construct_fedisic_train_val_datasets(
            self.client_number, str(self.data_path)
        )
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False)
        return train_loader, val_loader

    def get_model(self, config: Config) -> nn.Module:
        model: nn.Module = Baseline().to(self.device)
        return model

    def get_optimizer(self, config: Config) -> Dict[str, Optimizer]:
        # Note that the global optimizer operates on self.global_model.parameters() and
        global_optimizer = torch.optim.AdamW(self.global_model.parameters(), lr=self.learning_rate)
        local_optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        return {"global": global_optimizer, "local": local_optimizer}

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
        "--seed",
        action="store",
        type=int,
        help="Seed for the random number generators across python, torch, and numpy",
        required=False,
    )
    parser.add_argument(
        "--mu",
        action="store",
        type=float,
        help="Weight for the mkmmd losses",
        required=False,
    )
    parser.add_argument(
        "--deep_mmd_loss_depth",
        action="store",
        type=int,
        help="Depth of applying the deep mmd loss",
        required=False,
        default=1,
    )
    parser.add_argument(
        "--beta_update_interval",
        action="store",
        type=int,
        help="Interval for updating the beta values",
        required=False,
        default=20,
    )
    args = parser.parse_args()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(INFO, f"Device to be used: {DEVICE}")
    log(INFO, f"Server Address: {args.server_address}")
    log(INFO, f"Learning Rate: {args.learning_rate}")
    log(INFO, f"Mu: {args.mu}")
    log(INFO, f"DEEP MMD Loss Depth: {args.deep_mmd_loss_depth}")

    # Set the random seed for reproducibility
    set_all_random_seeds(args.seed)

    checkpoint_dir = os.path.join(args.artifact_dir, args.run_name)
    checkpoint_name = f"client_{args.client_number}_best_model.pkl"
    checkpointer = ClientCheckpointModule(post_aggregation=BestLossTorchCheckpointer(checkpoint_dir, checkpoint_name))

    client = FedIsic2019DittoClient(
        data_path=Path(args.dataset_dir),
        metrics=[BalancedAccuracy("FedIsic2019_balanced_accuracy")],
        device=DEVICE,
        client_number=args.client_number,
        learning_rate=args.learning_rate,
        checkpointer=checkpointer,
        deep_mmd_loss_depth=args.deep_mmd_loss_depth,
        deep_mmd_loss_weight=args.mu,
    )

    fl.client.start_client(server_address=args.server_address, client=client.to_client())

    # Shutdown the client gracefully
    client.shutdown()
