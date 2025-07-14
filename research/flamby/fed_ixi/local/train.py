import argparse
from logging import INFO

import torch
from flamby.datasets.fed_ixi import BATCH_SIZE, LR, NUM_EPOCHS_POOLED, Baseline, BaselineLoss
from flwr.common.logger import log
from torch import nn
from torch.utils.data import DataLoader

from fl4health.metrics import BinarySoftDiceCoefficient
from fl4health.metrics.metric_managers import MetricManager
from research.flamby.flamby_data_utils import construct_fed_ixi_train_val_datasets
from research.flamby.single_node_trainer import SingleNodeTrainer
from research.flamby.utils import summarize_model_info


class FedIxiLocalTrainer(SingleNodeTrainer):
    def __init__(
        self,
        device: torch.device,
        client_number: int,
        checkpoint_stub: str,
        dataset_dir: str,
        run_name: str = "",
    ) -> None:
        super().__init__(device, checkpoint_stub, dataset_dir, run_name)
        self.client_number = client_number

        train_dataset, validation_dataset = construct_fed_ixi_train_val_datasets(client_number, dataset_dir)

        self.train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        self.val_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # NOTE: We set the out_channels_first_layer to 12 rather than the default of 8. This roughly doubles the
        # size of the baseline model to be used (1106520 DOF). This is to allow for a fair parameter comparison with
        # FENDA and APFL
        self.model: nn.Module = Baseline(out_channels_first_layer=12).to(self.device)
        summarize_model_info(self.model)

        self.criterion = BaselineLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=LR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local Training Main")
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
        "--client_number",
        action="store",
        type=int,
        help="Number of the client for dataset loading (should be 0-2 for FedIXI)",
        required=True,
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(INFO, f"Device to be used: {device}")

    trainer = FedIxiLocalTrainer(
        device,
        args.client_number,
        args.artifact_dir,
        args.dataset_dir,
        args.run_name,
    )
    metrics = [BinarySoftDiceCoefficient("FedIXI_dice")]
    train_metric_mngr = MetricManager(metrics, "train_meter")
    val_metric_mngr = MetricManager(metrics, "val_meter")
    # Central and local models in FLamby for FedIXI are trained for 10 epochs
    trainer.train_by_epochs(NUM_EPOCHS_POOLED, train_metric_mngr, val_metric_mngr)
