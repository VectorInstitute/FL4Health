import argparse
import os
from logging import INFO
from pathlib import Path
from typing import Optional, Sequence, Tuple

import flwr as fl
import torch
import torch.nn as nn
from flamby.datasets.fed_ixi import BATCH_SIZE, LR, NUM_CLIENTS, Baseline, BaselineLoss
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
from fl4health.clients.partial_weight_exchange_client import PartialWeightExchangeClient
from fl4health.parameter_exchange.parameter_selection_criteria import largest_magnitude_change_scores
from fl4health.parameter_exchange.sparse_coo_parameter_exchanger import SparseCooParameterExchanger
from fl4health.utils.losses import LossMeterType
from fl4health.utils.metrics import BinarySoftDiceCoefficient, Metric
from fl4health.utils.random import set_all_random_seeds
from research.flamby.flamby_data_utils import construct_fed_ixi_train_val_datasets


class FedIxiSparseTensorClient(PartialWeightExchangeClient):
    def __init__(
        self,
        data_path: Path,
        metrics: Sequence[Metric],
        device: torch.device,
        client_number: int,
        learning_rate: float,
        sparsity_level: float,
        loss_meter_type: LossMeterType = LossMeterType.AVERAGE,
        checkpointer: Optional[TorchCheckpointer] = None,
        store_initial_model: bool = True,
    ) -> None:
        super().__init__(
            data_path=data_path,
            metrics=metrics,
            device=device,
            loss_meter_type=loss_meter_type,
            checkpointer=checkpointer,
            store_initial_model=store_initial_model,
        )
        self.client_number = client_number
        self.learning_rate: float = learning_rate
        self.sparsity_level = sparsity_level

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
        # NOTE: We set the out_channels_first_layer to 12 rather than the default of 8. This roughly doubles the size
        # of the baseline model to be used (1106520 DOF). This is to allow for a fair parameter comparison with FENDA
        # and APFL
        model: nn.Module = Baseline(out_channels_first_layer=12).to(self.device)
        return model

    def get_optimizer(self, config: Config) -> Optimizer:
        return torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)

    def get_criterion(self, config: Config) -> _Loss:
        return BaselineLoss()

    def get_parameter_exchanger(self, config: Config) -> SparseCooParameterExchanger:
        parameter_exchanger = SparseCooParameterExchanger(
            sparsity_level=self.sparsity_level,
            score_gen_function=largest_magnitude_change_scores,
        )
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
    parser.add_argument(
        "--sparsity_level",
        action="store",
        type=float,
        help="Level of sparsity used in partial weight exchange",
    )
    parser.add_argument(
        "--seed",
        action="store",
        type=int,
        help="Seed for the random number generators across python, torch, and numpy",
        required=False,
        default=47,
    )
    parser.add_argument(
        "--no_federated_checkpointing",
        action="store_true",
        help="boolean to disable client-side federated checkpointing in the personal FL experiment",
    )
    args = parser.parse_args()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(INFO, f"Device to be used: {DEVICE}")
    log(INFO, f"Server Address: {args.server_address}")
    log(INFO, f"Learning Rate: {args.learning_rate}")

    # Set the random seed for reproducibility
    set_all_random_seeds(args.seed)

    federated_checkpointing = not args.no_federated_checkpointing
    checkpoint_dir = os.path.join(args.artifact_dir, args.run_name)
    checkpoint_name = f"client_{args.client_number}_best_model.pkl"
    checkpointer = (
        BestMetricTorchCheckpointer(checkpoint_dir, checkpoint_name, maximize=False)
        if federated_checkpointing
        else LatestTorchCheckpointer(checkpoint_dir, checkpoint_name)
    )

    client = FedIxiSparseTensorClient(
        data_path=Path(args.dataset_dir),
        metrics=[BinarySoftDiceCoefficient("FedIXI_dice")],
        device=DEVICE,
        client_number=args.client_number,
        learning_rate=args.learning_rate,
        sparsity_level=args.sparsity_level,
        checkpointer=checkpointer,
    )

    fl.client.start_numpy_client(server_address=args.server_address, client=client)

    # Shutdown the client gracefully
    client.shutdown()