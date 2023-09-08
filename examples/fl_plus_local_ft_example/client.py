import argparse
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import flwr as fl
import torch
import torch.nn as nn
from flwr.common.typing import Config
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from examples.models.cnn_model import Net
from fl4health.checkpointing.checkpointer import TorchCheckpointer
from fl4health.clients.basic_client import BasicClient
from fl4health.utils.load_data import load_cifar10_data
from fl4health.utils.metrics import Accuracy, Metric


class CifarClient(BasicClient):
    def __init__(
        self,
        data_path: Path,
        metrics: Sequence[Metric],
        device: torch.device,
        meter_type: str = "average",
        use_wandb_reporter: bool = False,
        checkpointer: Optional[TorchCheckpointer] = None,
    ) -> None:
        super().__init__(
            data_path=data_path,
            metrics=metrics,
            device=device,
            meter_type=meter_type,
            use_wandb_reporter=use_wandb_reporter,
            checkpointer=checkpointer,
        )

    def get_data_loaders(self, config: Config, data_path: Path) -> Tuple[DataLoader, DataLoader]:
        batch_size = self.narrow_config_type(config, "batch_size", int)
        train_loader, val_loader, _ = load_cifar10_data(data_path, batch_size)
        return train_loader, val_loader

    def get_model(self, config: Config) -> nn.Module:
        model = Net().to(self.device)
        return model

    def get_optimizer(self, model: nn.Module, config: Config) -> Optimizer:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        return optimizer

    def compute_loss(self, preds: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        loss = torch.nn.functional.cross_entropy(preds, target)
        return loss, {}

    def predict(self, input: torch.Tensor) -> torch.Tensor:
        preds = self.model(input)
        return preds


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FL Client Main")
    parser.add_argument("--dataset_path", action="store", type=str, help="Path to the local dataset")
    args = parser.parse_args()

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_path = Path(args.dataset_path)
    metrics = [Accuracy("accuracy")]
    client = CifarClient(data_path, metrics, DEVICE)
    fl.client.start_numpy_client(server_address="0.0.0.0:8080", client=client)

    # Run further local training after the federated learning has finished
    client.train_by_epochs(2)
    # Finally, we evaluate the model
    client.validate()
