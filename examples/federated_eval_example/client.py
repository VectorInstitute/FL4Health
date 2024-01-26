import argparse
from pathlib import Path
from typing import Optional, Sequence, Tuple

import flwr as fl
import torch
import torch.nn as nn
from flwr.common.typing import Config
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader

from examples.models.cnn_model import Net
from fl4health.clients.evaluate_client import EvaluateClient
from fl4health.reporting.metrics import MetricsReporter
from fl4health.utils.load_data import load_cifar10_test_data
from fl4health.utils.losses import LossMeterType
from fl4health.utils.metrics import Accuracy, Metric


class CifarClient(EvaluateClient):
    def __init__(
        self,
        data_path: Path,
        metrics: Sequence[Metric],
        device: torch.device,
        model_checkpoint_path: Optional[Path],
        metrics_reporter: Optional[MetricsReporter] = None,
    ) -> None:
        super().__init__(
            data_path=data_path,
            metrics=metrics,
            device=device,
            model_checkpoint_path=model_checkpoint_path,
            loss_meter_type=LossMeterType.AVERAGE,
            metrics_reporter=metrics_reporter,
        )

    def initialize_global_model(self, config: Config) -> Optional[nn.Module]:
        # Initialized a global model to be hydrated with a server-side model if the parameters are passed
        return Net().to(self.device)

    def get_data_loader(self, config: Config) -> Tuple[DataLoader]:
        batch_size = self.narrow_config_type(config, "batch_size", int)
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

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    client = CifarClient(
        data_path=data_path,
        metrics=[Accuracy("accuracy")],
        device=DEVICE,
        model_checkpoint_path=client_checkpoint_path,
    )
    fl.client.start_numpy_client(server_address="0.0.0.0:8080", client=client)
    client.shutdown()
