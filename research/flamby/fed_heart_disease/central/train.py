import argparse
from logging import INFO
from typing import Tuple

import torch
import torch.nn as nn
from flamby.datasets.fed_heart_disease import BATCH_SIZE, LR, Baseline, BaselineLoss, FedHeartDisease
from flwr.common.logger import log
from torch.utils.data import DataLoader, random_split

from fl4health.utils.metrics import AccumulationMeter, Accuracy
from research.flamby.single_node_trainer import SingleNodeTrainer


class FedHeartDiseaseCentralizedTrainer(SingleNodeTrainer):
    def __init__(
        self,
        device: torch.device,
        checkpoint_stub: str,
        dataset_dir: str,
        run_name: str = "",
    ) -> None:
        super().__init__(device, checkpoint_stub, dataset_dir, run_name)

        train_dataset, validation_dataset = self.construct_train_val_datasets()

        self.train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        self.val_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False)

        self.model: nn.Module = Baseline().to(self.device)
        # NOTE: The class weights specified by alpha in this baseline loss are precomputed based on the weights of
        # the pool dataset. This is a bit of cheating but FLamby does it in their paper.
        self.criterion = BaselineLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=LR)

    def construct_train_val_datasets(self) -> Tuple[FedHeartDisease, FedHeartDisease]:
        full_train_dataset = FedHeartDisease(center=0, train=True, pooled=True, data_path=self.dataset_dir)
        # Something weird is happening with the typing of the split sequence in random split. Punting with a mypy
        # ignore for now.
        train_dataset, validation_dataset = tuple(random_split(full_train_dataset, [0.8, 0.2]))  # type: ignore
        return train_dataset, validation_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Centralized Training Main")
    parser.add_argument(
        "--artifact_dir",
        action="store",
        type=str,
        help="Path to save artifacts such as logs and model checkpoints",
        required=True,
    )
    parser.add_argument(
        "--dataset_dir",
        action="store",
        type=str,
        help="Path to the preprocessed Fed Heart Disease Dataset (ex. path/to/fed_heart_disease)",
        required=True,
    )
    parser.add_argument(
        "--run_name",
        action="store",
        help="Name of the run, model checkpoints will be saved under a subfolder with this name",
        required=True,
    )
    args = parser.parse_args()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(INFO, f"Device to be used: {DEVICE}")

    trainer = FedHeartDiseaseCentralizedTrainer(
        DEVICE,
        args.artifact_dir,
        args.dataset_dir,
        args.run_name,
    )
    metrics = [Accuracy("FedHeartDisease_accuracy")]
    train_meter = AccumulationMeter(metrics, "train_meter")
    val_meter = AccumulationMeter(metrics, "val_meter")
    # Central and local models in FLamby for Fed Heart Disease are trained for 20 epochs
    trainer.train_by_epochs(50, train_meter, val_meter)
