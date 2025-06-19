import argparse
import os
from collections.abc import Sequence
from logging import INFO
from pathlib import Path

import flwr as fl
import torch

# data and metrics
from data.data import load_train_delirium, load_train_mortality
from delirium_models.moon_model import DeliriumMoonModel
from flwr.common.logger import log
from flwr.common.typing import Config
from moon.moon_client import MoonClient

# Models
from mortality_models.moon_model import MortalityMoonModel
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from utils.random import set_all_random_seeds

from fl4health.checkpointing.checkpointer import BestMetricTorchCheckpointer, TorchModuleCheckpointer
from fl4health.metrics.base_metrics import Metric
from fl4health.utils.losses import LossMeterType
from research.gemini.metrics.metrics import Accuracy, BinaryF1, BinaryRocAuc


class GeminiMoonClient(MoonClient):
    def __init__(
        self,
        data_path: Path,
        metrics: Sequence[Metric],
        device: torch.device,
        hospital_id: list[str],
        learning_rate: float,
        learning_task: str,
        checkpoint_stub: str,
        run_name: str = "",
        loss_meter_type: LossMeterType = LossMeterType.AVERAGE,
        contrastive_weight: float = 10,
        checkpointer: TorchModuleCheckpointer | None = None,
    ) -> None:
        # Checkpointing: create a string of the names of the hospitals
        self.hospitals = hospital_id
        self.hospital_names = ",".join(self.hospitals)
        checkpoint_dir = os.path.join(checkpoint_stub, run_name)
        checkpoint_name = f"client_{self.hospital_names}_best_model.pkl"
        self.checkpointer = BestMetricTorchCheckpointer(checkpoint_dir, checkpoint_name, maximize=False)
        super().__init__(
            data_path=data_path,
            metrics=metrics,
            device=device,
            loss_meter_type=loss_meter_type,
            checkpointer=self.checkpointer,
            contrastive_weight=contrastive_weight,
        )
        self.learning_rate: float = learning_rate

        self.learning_task = learning_task
        self.learning_rate = learning_rate
        # Metrics initialization
        self.metrics = metrics

        log(INFO, f"Client Name: {self.client_name} Client hospitals {self.hospitals}")

    def get_data_loaders(self, config: Config) -> tuple[DataLoader, DataLoader]:
        batch_size = self.narrow_dict_type(config, "batch_size", int)
        if self.learning_task == "mortality":
            (
                train_loader,
                val_loader,
                num_examples,
            ) = load_train_mortality(self.data_path, batch_size, self.hospitals)
        else:
            train_loader, val_loader, num_examples = load_train_delirium(self.data_path, batch_size, self.hospitals)
        return train_loader, val_loader

    def get_model(self, config: Config) -> nn.Module:
        if self.learning_task == "mortality":
            model: nn.Module = MortalityMoonModel(input_dim=35, output_dim=1).to(self.device)
        else:
            model: nn.Module = DeliriumMoonModel(input_dim=8093, output_dim=1).to(self.device)
        return model

    def get_optimizer(self, config: Config) -> Optimizer:
        return torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)

    def get_criterion(self, config: Config) -> _Loss:
        return torch.nn.BCEWithLogitsLoss()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FL Client Main")
    parser.add_argument("--hospital_id", nargs="+", default=["THPC", "SMH"], help="ID of hospitals")
    parser.add_argument(
        "--task",
        action="store",
        type=str,
        default="mortality",
        help="GEMINI usecase: mortality, delirium",
    )

    parser.add_argument(
        "--artifact_dir",
        action="store",
        type=str,
        help="Path to save client artifacts such as logs and model checkpoints",
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
        "--learning_rate",
        action="store",
        type=float,
        help="Learning rate for local optimization",
        default=0.001,
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
        help="Weight for the contrastive loss",
        required=False,
    )

    args = parser.parse_args()
    # Define the path to the distributed data based on the task
    if args.task == "mortality":
        data_path = Path("mortality_data")
    elif args.task == "delirium":
        data_path = Path("delirium_data")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log(INFO, f"Device to be used: {device}")
    log(INFO, f"Task: {args.task}")
    log(INFO, f"Server Address: {args.server_address}")

    set_all_random_seeds(args.seed)

    client = GeminiMoonClient(
        data_path=data_path,
        metrics=[BinaryRocAuc(), BinaryF1(), Accuracy()],
        device=device,
        hospital_id=args.hospital_id,
        learning_rate=args.learning_rate,
        learning_task=args.task,
        checkpoint_stub=args.artifact_dir,
        run_name=args.run_name,
        contrastive_weight=args.mu,
    )
    fl.client.start_numpy_client(server_address=args.server_address, client=client)

    # Shutdown the client gracefully
    client.shutdown()
