import argparse
import os
from logging import INFO
from typing import Dict, Tuple

import torch
import torch.nn as nn
from flamby.datasets.fed_isic2019 import BATCH_SIZE, LR, Baseline, BaselineLoss, FedIsic2019
from flwr.common.logger import log
from flwr.common.typing import Scalar
from torch.utils.data import DataLoader, random_split

from fl4health.checkpointing.checkpointer import BestMetricTorchCheckpointer
from fl4health.utils.metrics import AccumulationMeter, BalancedAccuracy, Meter


class FedIsic2019LocalTrainer:
    def __init__(
        self,
        device: torch.device,
        client_number: int,
        checkpoint_stub: str,
        dataset_dir: str,
        run_name: str = "",
    ) -> None:
        self.device = device
        self.client_number = client_number
        checkpoint_dir = os.path.join(checkpoint_stub, run_name)
        # This is called the "server model" so that it can be found by the evaluate_on_holdout.py script
        checkpoint_name = "server_best_model.pkl"
        self.checkpointer = BestMetricTorchCheckpointer(checkpoint_dir, checkpoint_name, maximize=False)
        self.dataset_dir = dataset_dir

        train_dataset, validation_dataset = self.construct_train_val_datasets()

        self.train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        self.val_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False)

        self.model: nn.Module = Baseline().to(self.device)
        # NOTE: The class weights specified by alpha in this baseline loss are precomputed based on the weights of
        # the pool dataset. This is a bit of cheating but FLamby does it in their paper.
        self.criterion = BaselineLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=LR)

    def construct_train_val_datasets(self) -> Tuple[FedIsic2019, FedIsic2019]:
        full_train_dataset = FedIsic2019(
            center=self.client_number, train=True, pooled=False, data_path=self.dataset_dir
        )
        # Something weird is happening with the typing of the split sequence in random split. Punting with a mypy
        # ignore for now.
        train_dataset, validation_dataset = tuple(random_split(full_train_dataset, [0.8, 0.2]))  # type: ignore
        return train_dataset, validation_dataset

    def _maybe_checkpoint(self, comparison_metric: float) -> None:
        if self.checkpointer:
            self.checkpointer.maybe_checkpoint(self.model, comparison_metric)

    def _handle_reporting(
        self,
        loss: float,
        metrics_dict: Dict[str, Scalar],
        is_validation: bool = False,
    ) -> None:
        metric_string = "\t".join([f"{key}: {str(val)}" for key, val in metrics_dict.items()])
        metric_prefix = "Validation" if is_validation else "Training"
        log(
            INFO,
            f"Local {metric_prefix} Loss: {loss} \n" f"Local {metric_prefix} Metrics: {metric_string}",
        )

    def train_step(self, input: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # forward pass on the model
        preds = self.model(input)
        loss = self.criterion(preds, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss, preds

    def train_by_epochs(
        self,
        epochs: int,
        train_meter: Meter,
        val_meter: Meter,
    ) -> None:
        self.model.train()

        for local_epoch in range(epochs):
            train_meter.clear()
            running_loss = 0.0
            for input, target in self.train_loader:
                input, target = input.to(self.device), target.to(self.device)
                batch_loss, preds = self.train_step(input, target)
                running_loss += batch_loss.item()
                train_meter.update(preds, target)

            log(INFO, f"Local Epoch: {local_epoch}")
            running_loss = running_loss / len(self.train_loader)
            metrics = train_meter.compute()
            self._handle_reporting(running_loss, metrics)

            # After each epoch run a validation pass
            self.validate(val_meter)

    def validate(self, meter: Meter) -> None:
        self.model.eval()
        running_loss = 0.0
        meter.clear()

        with torch.no_grad():
            for input, target in self.val_loader:
                input, target = input.to(self.device), target.to(self.device)

                preds = self.model(input)
                batch_loss = self.criterion(preds, target)
                running_loss += batch_loss.item()
                meter.update(preds, target)

        running_loss = running_loss / len(self.val_loader)
        metrics = meter.compute()
        self._handle_reporting(running_loss, metrics, is_validation=True)
        self._maybe_checkpoint(running_loss)


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
        "--client_number",
        action="store",
        type=int,
        help="Number of the client for dataset loading (should be 0-5 for FedIsic2019)",
        required=True,
    )
    args = parser.parse_args()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(INFO, f"Device to be used: {DEVICE}")

    trainer = FedIsic2019LocalTrainer(
        DEVICE,
        args.client_number,
        args.artifact_dir,
        args.dataset_dir,
        args.run_name,
    )
    metrics = [BalancedAccuracy("FedIsic2019_balanced_accuracy")]
    train_meter = AccumulationMeter(metrics, "train_meter")
    val_meter = AccumulationMeter(metrics, "val_meter")
    trainer.train_by_epochs(20, train_meter, val_meter)
