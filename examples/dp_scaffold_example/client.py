import argparse
import warnings
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import flwr as fl
import torch
import torch.nn as nn
from flwr.common.typing import Config
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from examples.models.cnn_model import MnistNet
from fl4health.checkpointing.checkpointer import TorchCheckpointer
from fl4health.clients.scaffold_client import DPScaffoldClient
from fl4health.utils.load_data import load_mnist_data
from fl4health.utils.metrics import Accuracy, Metric
from fl4health.utils.sampler import DirichletLabelBasedSampler

warnings.filterwarnings("ignore", category=UserWarning)


class MnistDPScaffoldClient(DPScaffoldClient):
    def __init__(
        self,
        data_path: Path,
        metrics: Sequence[Metric],
        device: torch.device,
        learning_rate_local: float,
        meter_type: str = "average",
        use_wandb_reporter: bool = False,
        checkpointer: Optional[TorchCheckpointer] = None,
    ) -> None:
        super().__init__(
            data_path=data_path,
            metrics=metrics,
            device=device,
            learning_rate_local=learning_rate_local,
            meter_type=meter_type,
            use_wandb_reporter=use_wandb_reporter,
            checkpointer=checkpointer,
        )

    def get_data_loaders(self, config: Config) -> Tuple[DataLoader, DataLoader]:
        batch_size = self.narrow_config_type(config, "batch_size", int)
        sampler = DirichletLabelBasedSampler(list(range(10)), sample_percentage=1.0)
        train_loader, val_loader, _ = load_mnist_data(self.data_path, batch_size, sampler)
        return train_loader, val_loader

    def get_optimizer(self, config: Config) -> Optimizer:
        return torch.optim.SGD(self.model.parameters(), lr=self.learning_rate_local)

    def get_model(self, config: Config) -> nn.Module:
        return MnistNet().to(self.device)

    def compute_loss(self, preds: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        return torch.nn.functional.cross_entropy(preds, target), {}

    def predict(self, input: torch.Tensor) -> torch.Tensor:
        return self.model(input)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FL Client Main")
    parser.add_argument("--dataset_path", action="store", type=str, help="Path to the local dataset")

    args = parser.parse_args()

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_path = Path(args.dataset_path)

    client = MnistDPScaffoldClient(data_path=data_path, metrics=[Accuracy()], device=DEVICE, learning_rate_local=0.05)

    fl.client.start_numpy_client(server_address="0.0.0.0:8080", client=client)

    client.shutdown()
