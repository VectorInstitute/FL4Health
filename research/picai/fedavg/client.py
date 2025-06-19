import argparse
from collections.abc import Sequence
from logging import INFO
from pathlib import Path

import flwr as fl
import torch
from flwr.common.logger import log
from flwr.common.typing import Config
from monai.data.dataloader import DataLoader
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torchmetrics.classification import MultilabelAveragePrecision

from fl4health.checkpointing.client_module import ClientCheckpointAndStateModule
from fl4health.checkpointing.state_checkpointer import ClientStateCheckpointer
from fl4health.clients.basic_client import BasicClient
from fl4health.metrics import TorchMetric
from fl4health.metrics.base_metrics import Metric
from fl4health.reporting.base_reporter import BaseReporter
from fl4health.utils.losses import LossMeterType
from research.picai.data.data_utils import (
    get_dataloader,
    get_img_and_seg_paths,
    get_img_transform,
    get_seg_transform,
    split_img_and_seg_paths,
)
from research.picai.losses import FocalLoss
from research.picai.model_utils import get_model


torch.multiprocessing.set_sharing_strategy("file_system")


class PicaiFedAvgClient(BasicClient):
    def __init__(
        self,
        data_path: Path,
        metrics: Sequence[Metric],
        device: torch.device,
        loss_meter_type: LossMeterType = LossMeterType.AVERAGE,
        checkpoint_and_state_module: ClientCheckpointAndStateModule | None = None,
        reporters: Sequence[BaseReporter] | None = None,
        progress_bar: bool = False,
        client_name: str | None = None,
        overviews_dir: Path = Path("./"),
        data_partition: int | None = None,
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
        )

        self.data_partition = data_partition
        self.overviews_dir = overviews_dir
        self.class_proportions: torch.Tensor

    def get_data_loaders(self, config: Config) -> tuple[DataLoader, DataLoader]:
        train_img_paths, train_seg_paths, class_proportions = get_img_and_seg_paths(
            self.overviews_dir, int(config["fold_id"]), True
        )
        if self.data_partition is not None:
            assert isinstance(config["n_clients"], int)
            train_img_paths_list, train_seg_paths_list = split_img_and_seg_paths(
                train_img_paths, train_seg_paths, splits=config["n_clients"]
            )
            train_img_paths, train_seg_paths = (
                train_img_paths_list[self.data_partition],
                train_seg_paths_list[self.data_partition],
            )

        self.class_proportions = class_proportions
        train_loader = get_dataloader(
            train_img_paths,
            train_seg_paths,
            int(config["batch_size"]),
            get_img_transform(),
            get_seg_transform(),
            num_workers=2,
        )
        val_img_paths, val_seg_paths, _ = get_img_and_seg_paths(self.overviews_dir, int(config["fold_id"]), True)

        if self.data_partition is not None:
            val_img_paths_list, val_seg_paths_list = split_img_and_seg_paths(val_img_paths, val_seg_paths, 2)
            val_img_paths, val_seg_paths = (
                val_img_paths_list[self.data_partition],
                val_seg_paths_list[self.data_partition],
            )

        val_loader = get_dataloader(
            val_img_paths,
            val_seg_paths,
            int(config["batch_size"]),
            get_img_transform(),
            get_seg_transform(),
            num_workers=0,
        )
        return train_loader, val_loader

    def get_model(self, config: Config) -> nn.Module:
        return get_model(device=self.device)

    def get_criterion(self, config: Config) -> _Loss:
        return FocalLoss(alpha=self.class_proportions[-1].item())

    def get_optimizer(self, config: Config) -> Optimizer:
        return torch.optim.Adam(self.model.parameters(), amsgrad=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FL Client Main")
    parser.add_argument(
        "--artifact_dir",
        action="store",
        type=str,
        help="Path to dir to store run artifacts",
        required=True,
    )
    parser.add_argument(
        "--base_dir",
        action="store",
        type=str,
        help="Path to base directory containing PICAI dataset",
        required=True,
    )
    parser.add_argument(
        "--overviews_dir",
        action="store",
        type=str,
        help="Path to the directory containing the cross validation fold sheets",
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
        "--data_partition",
        type=int,
        help="The data partition to train the client on",
        default=0,
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(INFO, f"Device to be used: {device}")
    log(INFO, f"Server Address: {args.server_address}")

    metrics = [
        TorchMetric(
            name="MLAP",
            metric=MultilabelAveragePrecision(average="macro", num_labels=2, thresholds=3).to(device),
        )
    ]

    checkpoint_and_state_module: ClientCheckpointAndStateModule | None
    if args.artifact_dir is not None:
        checkpoint_and_state_module = ClientCheckpointAndStateModule(
            state_checkpointer=ClientStateCheckpointer(Path(args.artifact_dir))
        )
    else:
        checkpoint_and_state_module = None

    client = PicaiFedAvgClient(
        data_path=Path(args.base_dir),
        metrics=metrics,
        device=device,
        checkpoint_and_state_module=checkpoint_and_state_module,
        overviews_dir=args.overviews_dir,
        data_partition=args.data_partition,
    )

    fl.client.start_client(server_address=args.server_address, client=client.to_client())

    # Shutdown the client gracefully
    client.shutdown()
