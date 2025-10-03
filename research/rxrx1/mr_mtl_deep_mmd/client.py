import argparse
import os
from collections import OrderedDict
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
from torchvision import models

from fl4health.checkpointing.checkpointer import BestLossTorchModuleCheckpointer, LatestTorchModuleCheckpointer
from fl4health.checkpointing.client_module import ClientCheckpointAndStateModule
from fl4health.clients.deep_mmd_clients.mr_mtl_deep_mmd_client import MrMtlDeepMmdClient
from fl4health.datasets.rxrx1.load_data import load_rxrx1_data, load_rxrx1_test_data
from fl4health.metrics import Accuracy
from fl4health.metrics.base_metrics import Metric
from fl4health.reporting.base_reporter import BaseReporter
from fl4health.utils.config import narrow_dict_type
from fl4health.utils.losses import LossMeterType
from fl4health.utils.random import set_all_random_seeds


BASELINE_LAYERS: OrderedDict[str, int] = OrderedDict()
BASELINE_LAYERS["layer1"] = 1048576
BASELINE_LAYERS["layer2"] = 524288
BASELINE_LAYERS["layer3"] = 262144
BASELINE_LAYERS["layer4"] = 131072
BASELINE_LAYERS["avgpool"] = 512


class Rxrx1MrMtlClient(MrMtlDeepMmdClient):
    def __init__(
        self,
        data_path: Path,
        metrics: Sequence[Metric],
        device: torch.device,
        client_number: int,
        learning_rate: float,
        loss_meter_type: LossMeterType = LossMeterType.AVERAGE,
        deep_mmd_loss_weight: float = 10,
        deep_mmd_loss_depth: int = 1,
        checkpoint_and_state_module: ClientCheckpointAndStateModule | None = None,
        reporters: Sequence[BaseReporter] | None = None,
        progress_bar: bool = False,
        client_name: str | None = None,
    ) -> None:
        feature_extraction_layers_with_size = OrderedDict(list(BASELINE_LAYERS.items())[-1 * deep_mmd_loss_depth :])
        super().__init__(
            data_path=data_path,
            metrics=metrics,
            device=device,
            loss_meter_type=loss_meter_type,
            checkpoint_and_state_module=checkpoint_and_state_module,
            reporters=reporters,
            progress_bar=progress_bar,
            client_name=client_name,
            deep_mmd_loss_weight=deep_mmd_loss_weight,
            feature_extraction_layers_with_size=feature_extraction_layers_with_size,
        )
        self.client_number = client_number
        self.learning_rate: float = learning_rate

        log(INFO, f"Client Name: {self.client_name}, Client Number: {self.client_number}")

    def setup_client(self, config: Config) -> None:
        # Check if the client number is within the range of the total number of clients
        num_clients = narrow_dict_type(config, "n_clients", int)
        assert 0 <= self.client_number < num_clients
        super().setup_client(config)

    def get_data_loaders(self, config: Config) -> tuple[DataLoader, DataLoader]:
        batch_size = narrow_dict_type(config, "batch_size", int)
        train_loader, val_loader, _ = load_rxrx1_data(
            data_path=self.data_path, client_num=self.client_number, batch_size=batch_size, seed=self.client_number
        )

        return train_loader, val_loader

    def get_test_data_loader(self, config: Config) -> DataLoader | None:
        batch_size = narrow_dict_type(config, "batch_size", int)
        test_loader, _ = load_rxrx1_test_data(
            data_path=self.data_path, client_num=self.client_number, batch_size=batch_size
        )

        return test_loader

    def get_criterion(self, config: Config) -> _Loss:
        return torch.nn.CrossEntropyLoss()

    def get_optimizer(self, config: Config) -> Optimizer:
        return torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)

    def get_model(self, config: Config) -> nn.Module:
        model = models.resnet18(pretrained=True)
        num_classes = 50
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model.to(self.device)


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
        help="Number of the client for dataset loading (should be 0-3 for Rxrx1)",
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
        "--mu",
        action="store",
        type=float,
        help="Weight for the Deep MMD losses",
        required=False,
    )
    parser.add_argument(
        "--deep_mmd_loss_depth",
        action="store",
        type=int,
        help="Depth of applying the Deep MMD loss",
        required=False,
        default=1,
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(INFO, f"Device to be used: {device}")
    log(INFO, f"Server Address: {args.server_address}")
    log(INFO, f"Learning Rate: {args.learning_rate}")
    log(INFO, f"Mu: {args.mu}")
    log(INFO, f"DEEP MMD Loss Depth: {args.deep_mmd_loss_depth}")

    # Set the random seed for reproducibility
    set_all_random_seeds(args.seed)

    # Adding extensive checkpointing for the client
    checkpoint_dir = os.path.join(args.artifact_dir, args.run_name)
    pre_aggregation_best_checkpoint_name = f"pre_aggregation_client_{args.client_number}_best_model.pkl"
    pre_aggregation_last_checkpoint_name = f"pre_aggregation_client_{args.client_number}_last_model.pkl"
    post_aggregation_best_checkpoint_name = f"post_aggregation_client_{args.client_number}_best_model.pkl"
    post_aggregation_last_checkpoint_name = f"post_aggregation_client_{args.client_number}_last_model.pkl"
    checkpoint_and_state_module = ClientCheckpointAndStateModule(
        pre_aggregation=[
            BestLossTorchModuleCheckpointer(checkpoint_dir, pre_aggregation_best_checkpoint_name),
            LatestTorchModuleCheckpointer(checkpoint_dir, pre_aggregation_last_checkpoint_name),
        ],
        post_aggregation=[
            BestLossTorchModuleCheckpointer(checkpoint_dir, post_aggregation_best_checkpoint_name),
            LatestTorchModuleCheckpointer(checkpoint_dir, post_aggregation_last_checkpoint_name),
        ],
    )

    data_path = Path(args.dataset_dir)
    client = Rxrx1MrMtlClient(
        data_path=data_path,
        metrics=[Accuracy("accuracy")],
        device=device,
        client_number=args.client_number,
        learning_rate=args.learning_rate,
        checkpoint_and_state_module=checkpoint_and_state_module,
        deep_mmd_loss_depth=args.deep_mmd_loss_depth,
        deep_mmd_loss_weight=args.mu,
    )

    fl.client.start_client(server_address=args.server_address, client=client.to_client())
    # Shutdown the client gracefully
    client.shutdown()
