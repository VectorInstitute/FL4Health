import argparse
from collections.abc import Sequence
from pathlib import Path

import flwr as fl
import torch
from flwr.common.typing import Config
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader

from examples.models.cnn_model import Net
from fl4health.clients.evaluate_client import EvaluateClient
from fl4health.metrics import Accuracy
from fl4health.metrics.base_metrics import Metric
from fl4health.reporting.base_reporter import BaseReporter
from fl4health.utils.config import narrow_dict_type
from fl4health.utils.load_data import load_cifar10_test_data
from fl4health.utils.losses import LossMeterType


class CifarClient(EvaluateClient):
    def __init__(
        self,
        data_path: Path,
        metrics: Sequence[Metric],
        device: torch.device,
        model_checkpoint_path: Path | None,
        reporters: Sequence[BaseReporter] | None = None,
    ) -> None:
        super().__init__(
            data_path=data_path,
            metrics=metrics,
            device=device,
            model_checkpoint_path=model_checkpoint_path,
            loss_meter_type=LossMeterType.AVERAGE,
            reporters=reporters,
        )

    def initialize_global_model(self, config: Config) -> nn.Module | None:
        # Initialized a global model to be hydrated with a server-side model if the parameters are passed
        return Net().to(self.device)

    def get_data_loader(self, config: Config) -> tuple[DataLoader]:
        batch_size = narrow_dict_type(config, "batch_size", int)
        evaluation_loader, _ = load_cifar10_test_data(self.data_path, batch_size)
        return (evaluation_loader,)

    def get_criterion(self, config: Config) -> _Loss:
        return torch.nn.CrossEntropyLoss()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FL Client Main")
    parser.add_argument("--dataset_path", action="store", type=str, help="Path to the local dataset")
    parser.add_argument(
        "--checkpoint_path",
        action="store",
        type=str,
        help="Path to client model checkpoint.",
        required=False,
    )
    args = parser.parse_args()
    data_path = Path(args.dataset_path)
    client_checkpoint_path = Path(args.checkpoint_path) if args.checkpoint_path else None

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    client = CifarClient(
        data_path=data_path,
        metrics=[Accuracy("accuracy")],
        device=device,
        model_checkpoint_path=client_checkpoint_path,
    )
    fl.client.start_client(server_address="0.0.0.0:8080", client=client.to_client())
    client.shutdown()
