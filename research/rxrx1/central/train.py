import argparse
import copy
from logging import INFO
from pathlib import Path

import torch
from flwr.common.logger import log
from torch import nn
from torch.utils.data import DataLoader

from fl4health.datasets.rxrx1.load_data import load_rxrx1_data
from fl4health.metrics import Accuracy
from fl4health.metrics.metric_managers import MetricManager
from fl4health.utils.dataset import TensorDataset
from research.rxrx1.single_node_trainer import SingleNodeTrainer
from research.rxrx1.utils import get_model


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
            assert isinstance(train_loader.dataset, TensorDataset), "Expected TensorDataset."
            assert isinstance(val_loader.dataset, TensorDataset), "Expected TensorDataset."

            if client_number == 0:
                aggregated_train_dataset = copy.deepcopy(train_loader.dataset)
                aggregated_val_dataset = copy.deepcopy(val_loader.dataset)
            else:
                assert aggregated_train_dataset.data is not None
                aggregated_train_dataset.data = torch.cat((aggregated_train_dataset.data, train_loader.dataset.data))
                assert train_loader.dataset.targets is not None and aggregated_train_dataset.targets is not None
                aggregated_train_dataset.targets = torch.cat(
                    (aggregated_train_dataset.targets, train_loader.dataset.targets)
                )

                assert aggregated_val_dataset.data is not None
                aggregated_val_dataset.data = torch.cat((aggregated_val_dataset.data, val_loader.dataset.data))
                assert val_loader.dataset.targets is not None and aggregated_val_dataset.targets is not None
                aggregated_val_dataset.targets = torch.cat(
                    (aggregated_val_dataset.targets, val_loader.dataset.targets)
                )

            log(INFO, f"Aggregated train dataset size: {len(aggregated_train_dataset.data)}")
            log(INFO, f"Aggregated val dataset size: {len(aggregated_val_dataset.data)}")

        self.train_loader = DataLoader(aggregated_train_dataset, batch_size=32, shuffle=True)
        self.val_loader = DataLoader(aggregated_val_dataset, batch_size=32, shuffle=False)

        self.model: nn.Module = get_model()
        self.model.to(self.device)
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
        help="Path to the preprocessed Rxrx1 Dataset",
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
