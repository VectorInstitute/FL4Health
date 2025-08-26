import argparse
import os
from collections.abc import Sequence
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

from fl4health.checkpointing.checkpointer import BestLossTorchModuleCheckpointer, LatestTorchModuleCheckpointer
from fl4health.checkpointing.client_module import ClientCheckpointAndStateModule
from fl4health.clients.mkmmd_clients.mr_mtl_mkmmd_client import MrMtlMkMmdClient
from fl4health.metrics import Accuracy
from fl4health.metrics.base_metrics import Metric
from fl4health.reporting.base_reporter import BaseReporter
from fl4health.utils.config import narrow_dict_type
from fl4health.utils.losses import LossMeterType
from fl4health.utils.random import set_all_random_seeds
from research.synthetic_data.model import FullyConnectedNet
from research.synthetic_data.preprocess import get_preprocessed_data, get_test_preprocessed_data


BASELINE_LAYERS = ["linear_1"]


class SyntheticMrMrlMmdClient(MrMtlMkMmdClient):
    def __init__(
        self,
        data_path: Path,
        metrics: Sequence[Metric],
        device: torch.device,
        client_number: int,
        learning_rate: float,
        heterogeneity_level: float,
        loss_meter_type: LossMeterType = LossMeterType.AVERAGE,
        mkmmd_loss_weight: float = 10,
        feature_l2_norm_weight: float = 1,
        mkmmd_loss_depth: int = 1,
        beta_global_update_interval: int = 20,
        checkpoint_and_state_module: ClientCheckpointAndStateModule | None = None,
        reporters: Sequence[BaseReporter] | None = None,
        progress_bar: bool = False,
        client_name: str | None = None,
        num_accumulating_batches: int | None = 50,
    ) -> None:
        super().__init__(
            data_path=data_path,
            metrics=metrics,
            device=device,
            loss_meter_type=loss_meter_type,
            checkpoint_and_state_module=checkpoint_and_state_module,
            reporters=reporters,
            progress_bar=progress_bar,
            client_name=client_name,
            mkmmd_loss_weight=mkmmd_loss_weight,
            feature_extraction_layers=BASELINE_LAYERS[-1 * mkmmd_loss_depth :],
            feature_l2_norm_weight=feature_l2_norm_weight,
            beta_global_update_interval=beta_global_update_interval,
            num_accumulating_batches=num_accumulating_batches,
        )
        self.client_number = client_number
        self.heterogeneity_level = heterogeneity_level
        self.learning_rate: float = learning_rate

    def setup_client(self, config: Config) -> None:
        # Check if the client number is within the range of the total number of clients
        num_clients = narrow_dict_type(config, "n_clients", int)
        assert 0 <= self.client_number < num_clients
        super().setup_client(config)

    def get_data_loaders(self, config: Config) -> tuple[DataLoader, DataLoader]:
        batch_size = narrow_dict_type(config, "batch_size", int)
        # The partitioned data should be generated prior to running the clients via preprocess_data function
        # in the research/synthetic_data/preprocess.py file
        train_loader, val_loader, _ = get_preprocessed_data(
            self.data_path, self.client_number, batch_size, self.heterogeneity_level, self.heterogeneity_level
        )
        return train_loader, val_loader

    def get_test_data_loader(self, config: Config) -> DataLoader | None:
        batch_size = narrow_dict_type(config, "batch_size", int)
        # The partitioned data should be generated prior to running the clients via preprocess_data function
        # in the research/synthetic_data/preprocess.py file
        test_loader, _ = get_test_preprocessed_data(
            self.data_path,
            self.client_number,
            batch_size,
            self.heterogeneity_level,
            self.heterogeneity_level,
        )
        return test_loader

    def get_criterion(self, config: Config) -> _Loss:
        return torch.nn.CrossEntropyLoss()

    def get_optimizer(self, config: Config) -> Optimizer:
        return torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=0.001)

    def get_model(self, config: Config) -> nn.Module:
        return FullyConnectedNet().to(self.device)


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
        help="Path to the preprocessed Synthetic Dataset",
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
        help="Number of the client for dataset loading",
        required=True,
    )
    parser.add_argument(
        "--learning_rate", action="store", type=float, help="Learning rate for local optimization", default=0.1
    )
    parser.add_argument(
        "--seed",
        action="store",
        type=int,
        help="Seed for the random number generators across python, torch, and numpy",
        required=False,
    )
    parser.add_argument(
        "--alpha_beta",
        action="store",
        type=float,
        help="Heterogeneity level for the dataset",
        required=True,
    )
    parser.add_argument(
        "--mu",
        action="store",
        type=float,
        help="Weight for the mkmmd losses",
        required=False,
    )
    parser.add_argument(
        "--l2",
        action="store",
        type=float,
        help="Weight for the feature l2 norm loss as a regularizer",
        required=False,
    )
    parser.add_argument(
        "--mkmmd_loss_depth",
        action="store",
        type=int,
        help="Depth of applying the mkmmd loss",
        required=False,
        default=1,
    )
    parser.add_argument(
        "--beta_update_interval",
        action="store",
        type=int,
        help="Interval for updating the beta values of mkmmd loss",
        required=False,
        default=20,
    )
    args = parser.parse_args()
    log(INFO, "Using preprocessed partitioned data for training, validation and testing")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(INFO, f"Device to be used: {device}")
    log(INFO, f"Server Address: {args.server_address}")
    log(INFO, f"Learning Rate: {args.learning_rate}")
    log(INFO, f"Mu: {args.mu}")
    log(INFO, f"Feature L2 Norm Weight: {args.l2}")
    log(INFO, f"MKMMD Loss Depth: {args.mkmmd_loss_depth}")
    log(INFO, f"Beta Update Interval: {args.beta_update_interval}")

    # Set the random seed for reproducibility
    set_all_random_seeds(args.seed)

    # Adding extensive checkpointing for the client
    checkpoint_dir = os.path.join(args.artifact_dir, args.run_name)
    pre_aggregation_best_checkpoint_name = f"pre_aggregation_client_{args.client_number}_best_model.pkl"
    pre_aggregation_last_checkpoint_name = f"pre_aggregation_client_{args.client_number}_last_model.pkl"
    checkpoint_and_state_module = ClientCheckpointAndStateModule(
        pre_aggregation=[
            BestLossTorchModuleCheckpointer(checkpoint_dir, pre_aggregation_best_checkpoint_name),
            LatestTorchModuleCheckpointer(checkpoint_dir, pre_aggregation_last_checkpoint_name),
        ],
    )

    data_path = Path(args.dataset_dir)
    client = SyntheticMrMrlMmdClient(
        data_path=data_path,
        metrics=[Accuracy("accuracy")],
        device=device,
        client_number=args.client_number,
        learning_rate=args.learning_rate,
        heterogeneity_level=args.alpha_beta,
        checkpoint_and_state_module=checkpoint_and_state_module,
        feature_l2_norm_weight=args.l2,
        mkmmd_loss_depth=args.mkmmd_loss_depth,
        mkmmd_loss_weight=args.mu,
        beta_global_update_interval=args.beta_update_interval,
    )

    fl.client.start_client(server_address=args.server_address, client=client.to_client())
    client.shutdown()
