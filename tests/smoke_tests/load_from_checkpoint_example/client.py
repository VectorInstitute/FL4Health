import argparse
from collections.abc import Sequence
from pathlib import Path

import flwr as fl
import torch
from flwr.common.typing import Config, NDArrays, Scalar
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from examples.models.cnn_model import Net
from fl4health.checkpointing.client_module import ClientCheckpointAndStateModule
from fl4health.checkpointing.state_checkpointer import ClientStateCheckpointer
from fl4health.clients.basic_client import BasicClient
from fl4health.metrics import Accuracy
from fl4health.metrics.base_metrics import Metric
from fl4health.reporting import JsonReporter
from fl4health.reporting.base_reporter import BaseReporter
from fl4health.utils.config import narrow_dict_type
from fl4health.utils.load_data import load_cifar10_data, load_cifar10_test_data
from fl4health.utils.losses import LossMeterType
from fl4health.utils.random import set_all_random_seeds


class CifarClient(BasicClient):
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
        seed: int = 42,
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
        self.seed = seed

    def get_data_loaders(self, config: Config) -> tuple[DataLoader, DataLoader]:
        batch_size = narrow_dict_type(config, "batch_size", int)
        train_loader, val_loader, _ = load_cifar10_data(self.data_path, batch_size)
        return train_loader, val_loader

    def get_test_data_loader(self, config: Config) -> DataLoader | None:
        batch_size = narrow_dict_type(config, "batch_size", int)
        test_loader, _ = load_cifar10_test_data(self.data_path, batch_size)
        return test_loader

    def get_criterion(self, config: Config) -> _Loss:
        return torch.nn.CrossEntropyLoss()

    def get_optimizer(self, config: Config) -> Optimizer:
        return torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

    def get_model(self, config: Config) -> nn.Module:
        return Net().to(self.device)

    def fit(self, parameters: NDArrays, config: Config) -> tuple[NDArrays, int, dict[str, Scalar]]:
        set_all_random_seeds(self.seed)
        return super().fit(parameters, config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FL Client Main")
    parser.add_argument("--dataset_path", action="store", type=str, help="Path to the local dataset")
    parser.add_argument(
        "--intermediate_client_state_dir",
        action="store",
        type=str,
        help="Path to intermediate checkpoint directory.",
        default="./",
    )
    parser.add_argument(
        "--client_name",
        action="store",
        type=str,
        help="Unique string identifier for client",
    )
    parser.add_argument(
        "--seed",
        action="store",
        type=int,
        help="Seed for the random number generators across python, torch, and numpy",
        required=False,
    )
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_path = Path(args.dataset_path)

    # Set the random seed for reproducibility
    set_all_random_seeds(args.seed)

    checkpoint_and_state_module: ClientCheckpointAndStateModule | None
    if args.intermediate_client_state_dir is not None:
        checkpoint_and_state_module = ClientCheckpointAndStateModule(
            state_checkpointer=ClientStateCheckpointer(Path(args.intermediate_client_state_dir))
        )
    else:
        checkpoint_and_state_module = None

    client = CifarClient(
        data_path,
        [Accuracy("accuracy")],
        device,
        checkpoint_and_state_module=checkpoint_and_state_module,
        client_name=args.client_name,
        seed=args.seed,
        reporters=[JsonReporter()],
    )
    fl.client.start_client(server_address="0.0.0.0:8080", client=client.to_client())

    client.shutdown()
