import argparse
from logging import INFO

import torch
from flamby.datasets.fed_isic2019 import BATCH_SIZE, LR, NUM_EPOCHS_POOLED, Baseline, BaselineLoss
from flwr.common.logger import log
from torch import nn
from torch.utils.data import DataLoader

from fl4health.metrics import BalancedAccuracy
from fl4health.metrics.metric_managers import MetricManager
from research.flamby.flamby_data_utils import construct_fedisic_train_val_datasets
from research.flamby.single_node_trainer import SingleNodeTrainer


class FedIsic2019CentralizedTrainer(SingleNodeTrainer):
    def __init__(
        self,
        device: torch.device,
        checkpoint_stub: str,
        dataset_dir: str,
        run_name: str = "",
    ) -> None:
        super().__init__(device, checkpoint_stub, dataset_dir, run_name)

        train_dataset, validation_dataset = construct_fedisic_train_val_datasets(
            client_number=0, dataset_dir=dataset_dir, pooled=True
        )

        self.train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        self.val_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False)

        self.model: nn.Module = Baseline().to(self.device)
        # NOTE: The class weights specified by alpha in this baseline loss are precomputed based on the weights of
        # the pool dataset. This is a bit of cheating but FLamby does it in their paper.
        self.criterion = BaselineLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=LR)


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
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(INFO, f"Device to be used: {device}")

    trainer = FedIsic2019CentralizedTrainer(
        device,
        args.artifact_dir,
        args.dataset_dir,
        args.run_name,
    )
    metrics = [BalancedAccuracy("FedIsic2019_balanced_accuracy")]
    train_metric_mngr = MetricManager(metrics, "train_meter")
    val_metric_mngr = MetricManager(metrics, "val_meter")
    # Central and local models in FLamby for FedISic are trained for 20 epochs
    trainer.train_by_epochs(NUM_EPOCHS_POOLED, train_metric_mngr, val_metric_mngr)
