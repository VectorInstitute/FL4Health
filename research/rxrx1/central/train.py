import argparse
from logging import INFO
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from flwr.common.logger import log
from torch.utils.data import DataLoader, Subset
from torchvision import models

from fl4health.datasets.rxrx1.dataset import Rxrx1Dataset
from fl4health.datasets.rxrx1.load_data import load_rxrx1_data
from fl4health.utils.metrics import Accuracy, MetricManager
from research.rxrx1.single_node_trainer import SingleNodeTrainer

NUM_CLIENTS = 4
EPOCHS = 100


class Rxrx1CentralizedTrainer(SingleNodeTrainer):
    def __init__(
        self,
        device: torch.device,
        checkpoint_stub: str,
        dataset_dir: str,
        run_name: str = "",
        lr: float = 0.001,
    ) -> None:
        super().__init__(device, checkpoint_stub, dataset_dir, run_name)

        for client_number in range(NUM_CLIENTS):
            train_loader, val_loader, num_examples = load_rxrx1_data(
                data_path=Path(dataset_dir), client_num=client_number, batch_size=32
            )
            assert isinstance(train_loader.dataset, Subset), "Expected Subset."
            assert isinstance(val_loader.dataset, Subset), "Expected Subset."
            assert isinstance(train_loader.dataset.dataset, Rxrx1Dataset), "Expected Rxrx1Dataset."
            assert isinstance(val_loader.dataset.dataset, Rxrx1Dataset), "Expected Rxrx1Dataset."

            if client_number == 0:
                meta_data_train = train_loader.dataset.dataset.metadata
                meta_data_val = val_loader.dataset.dataset.metadata
            else:
                meta_data_train = pd.concat([meta_data_train, train_loader.dataset.dataset.metadata])
                meta_data_val = pd.concat([meta_data_val, val_loader.dataset.dataset.metadata])

        aggregated_train_dataset = Rxrx1Dataset(
            metadata=meta_data_train, root=Path(dataset_dir), dataset_type="train", transform=None
        )
        aggregated_val_dataset = Rxrx1Dataset(
            metadata=meta_data_val, root=Path(dataset_dir), dataset_type="train", transform=None
        )

        self.train_loader = DataLoader(aggregated_train_dataset, batch_size=32, shuffle=True)
        self.val_loader = DataLoader(aggregated_val_dataset, batch_size=32, shuffle=False)

        self.model: nn.Module = models.resnet18(pretrained=True).to(self.device)
        # NOTE: The class weights specified by alpha in this baseline loss are precomputed based on the weights of
        # the pool dataset. This is a bit of cheating but FLamby does it in their paper.
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)


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
        "--lr",
        action="store",
        help="Learning rate for the optimizer",
        required=True,
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(INFO, f"Device to be used: {device}")

    trainer = Rxrx1CentralizedTrainer(
        device,
        args.artifact_dir,
        args.dataset_dir,
        args.run_name,
    )
    metrics = [Accuracy("Rxrx1_accuracy")]
    train_metric_mngr = MetricManager(metrics, "train_meter")
    val_metric_mngr = MetricManager(metrics, "val_meter")
    # Central and local models in FLamby for FedISic are trained for 20 epochs
    trainer.train_by_epochs(EPOCHS, train_metric_mngr, val_metric_mngr)
