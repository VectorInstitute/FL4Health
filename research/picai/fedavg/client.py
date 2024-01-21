import argparse
from logging import INFO
import torch
import torch.nn as nn
from torchmetrics.classification import MultilabelAveragePrecision
from torch.optim import Optimizer
from typing import Sequence, Tuple
from pathlib import Path
import flwr as fl
from flwr.common.logger import log
from flwr.common.typing import Config
from monai.data.dataloader import DataLoader

from fl4health.utils.metrics import Metric
from fl4health.utils.metrics import TorchMetric
from fl4health.clients.basic_client import BasicClient

from research.picai.losses import FocalLoss
from research.picai.model_utils import get_model
from research.picai.data_utils import get_dataloader, get_img_and_seg_paths, get_img_transform, get_seg_transform


class PicaiFedAvgClient(BasicClient):
    def __init__(
        self,
        data_path: Path,
        metrics: Sequence[Metric],
        device: torch.device,
        overviews_dir: Path
    ) -> None:

        super().__init__(data_path, metrics, device)
        self.overviews_dir = overviews_dir

    def get_data_loaders(self, config: Config) -> Tuple[DataLoader, DataLoader]:
        train_img_paths, train_seg_paths, _ = get_img_and_seg_paths(
            self.overviews_dir, self.data_path, int(config["fold_id"]), True)
        train_loader = get_dataloader(train_img_paths, train_seg_paths, int(
            config["batch_size"]), get_img_transform(), get_seg_transform())
        val_img_paths, val_seg_paths, _ = get_img_and_seg_paths(
            self.overviews_dir, self.data_path, int(config["fold_id"]), True)
        val_loader = get_dataloader(val_img_paths, val_seg_paths, int(
            config["batch_size"]), get_img_transform(), get_seg_transform())
        return train_loader, val_loader

    def get_model(self, config: Config) -> nn.Module:
        return get_model(device=self.device)

    def get_criterion(self, config: Config):
        return FocalLoss(alpha=0.75)

    def get_optimizer(self, config: Config) -> Optimizer:
        return torch.optim.Adam(self.model.parameters(), amsgrad=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FL Client Main")
    parser.add_argument(
        "--dataset_dir",
        action="store",
        type=str,
        help="Path to the preprocessed PICAI Dataset (ex. path/to/fedisic2019)",
        default="/ssd003/projects/aieng/public/PICAI/",
    )
    parser.add_argument(
        "--overviews_dir",
        action="store",
        type=str,
        help="Path to the directory containing the cross validation fold sheets",
        default="/ssd003/projects/aieng/public/PICAI/workdir/results/UNet/overviews/Task2203_picai_baseline/",
    )
    parser.add_argument(
        "--server_address",
        action="store",
        type=str,
        help="Server Address for the clients to communicate with the server through",
        default="0.0.0.0:8080",
    )
    args = parser.parse_args()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(INFO, f"Device to be used: {DEVICE}")
    log(INFO, f"Server Address: {args.server_address}")

    metrics = [TorchMetric(name="MLAP", metric=MultilabelAveragePrecision(
        average="macro", num_labels=2, thresholds=3).to(DEVICE))]

    client = PicaiFedAvgClient(
        data_path=Path(args.dataset_dir),
        metrics=metrics,
        device=DEVICE,
        overviews_dir=args.overviews_dir
    )

    fl.client.start_numpy_client(server_address=args.server_address, client=client)

    # Shutdown the client gracefully
    client.shutdown()
