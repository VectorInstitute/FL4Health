import argparse
import os
from logging import INFO
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import flwr as fl
import torch
from flwr.common.logger import log
from flwr.common.typing import Config
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from fl4health.checkpointing.checkpointer import BestLossTorchCheckpointer, LatestTorchCheckpointer
from fl4health.checkpointing.client_module import ClientCheckpointModule
from fl4health.clients.fenda_ditto_client import FendaDittoClient
from fl4health.model_bases.fenda_base import FendaModel
from fl4health.model_bases.sequential_split_models import SequentiallySplitModel
from fl4health.utils.config import narrow_dict_type
from fl4health.utils.losses import LossMeterType
from fl4health.utils.metrics import F1, Accuracy, Metric
from fl4health.utils.random import set_all_random_seeds
from research.cifar10.model import ConvNetFendaDittoGlobalModel, ConvNetFendaModel
from research.cifar10.preprocess import get_preprocessed_data


class CifarFendaDittoClient(FendaDittoClient):
    def __init__(
        self,
        data_path: Path,
        metrics: Sequence[Metric],
        device: torch.device,
        client_number: int,
        learning_rate: float,
        heterogeneity_level: float,
        loss_meter_type: LossMeterType = LossMeterType.AVERAGE,
        checkpointer: Optional[ClientCheckpointModule] = None,
        freeze_global_feature_extractor: bool = False,
    ) -> None:
        super().__init__(
            data_path=data_path,
            metrics=metrics,
            device=device,
            loss_meter_type=loss_meter_type,
            checkpointer=checkpointer,
            freeze_global_feature_extractor=freeze_global_feature_extractor,
        )
        self.client_number = client_number
        self.heterogeneity_level = heterogeneity_level
        self.learning_rate: float = learning_rate

        log(INFO, f"Client Name: {self.client_name}, Client Number: {self.client_number}")

    def get_data_loaders(self, config: Config) -> Tuple[DataLoader, DataLoader]:
        batch_size = narrow_dict_type(config, "batch_size", int)
        train_loader, val_loader, _ = get_preprocessed_data(
            self.data_path, self.client_number, batch_size, self.heterogeneity_level
        )
        return train_loader, val_loader

    def get_criterion(self, config: Config) -> _Loss:
        return torch.nn.CrossEntropyLoss()

    def get_optimizer(self, config: Config) -> Dict[str, Optimizer]:
        global_optimizer = torch.optim.AdamW(self.global_model.parameters(), lr=self.learning_rate)
        local_optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        return {"global": global_optimizer, "local": local_optimizer}

    def get_model(self, config: Config) -> FendaModel:
        return ConvNetFendaModel(in_channels=3, use_bn=False, dropout=0.1, hidden=512).to(self.device)

    def get_global_model(self, config: Config) -> SequentiallySplitModel:
        return ConvNetFendaDittoGlobalModel(in_channels=3, use_bn=False, dropout=0.1, hidden=512).to(self.device)


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
        help="Path to the preprocessed Cifar 10 Dataset",
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
        help="Number of the client for dataset loading",
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
        "--beta",
        action="store",
        type=float,
        help="Heterogeneity level for the dataset",
        required=True,
    )
    parser.add_argument(
        "--freeze_global_extractor",
        action="store_true",
        help="Whether or not to freeze the global feature extractor of the FENDA model or not.",
        default=False,
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(INFO, f"Device to be used: {device}")
    log(INFO, f"Server Address: {args.server_address}")
    log(INFO, f"Learning Rate: {args.learning_rate}")
    log(INFO, f"Beta: {args.beta}")
    if args.freeze_global_extractor:
        log(INFO, "Freezing the global feature extractor of the FENDA model")

    # Set the random seed for reproducibility
    set_all_random_seeds(args.seed, use_deterministic_torch_algos=True, disable_torch_benchmarking=True)

    # Adding extensive checkpointing for the client
    checkpoint_dir = os.path.join(args.artifact_dir, args.run_name)
    pre_aggregation_best_checkpoint_name = f"pre_aggregation_client_{args.client_number}_best_model.pkl"
    pre_aggregation_last_checkpoint_name = f"pre_aggregation_client_{args.client_number}_last_model.pkl"
    post_aggregation_best_checkpoint_name = f"post_aggregation_client_{args.client_number}_best_model.pkl"
    post_aggregation_last_checkpoint_name = f"post_aggregation_client_{args.client_number}_last_model.pkl"
    checkpointer = ClientCheckpointModule(
        pre_aggregation=[
            BestLossTorchCheckpointer(checkpoint_dir, pre_aggregation_best_checkpoint_name),
            LatestTorchCheckpointer(checkpoint_dir, pre_aggregation_last_checkpoint_name),
        ],
        post_aggregation=[
            BestLossTorchCheckpointer(checkpoint_dir, post_aggregation_best_checkpoint_name),
            LatestTorchCheckpointer(checkpoint_dir, post_aggregation_last_checkpoint_name),
        ],
    )

    data_path = Path(args.dataset_dir)
    client = CifarFendaDittoClient(
        data_path=data_path,
        metrics=[
            Accuracy("accuracy"),
            F1("f1_score_macro", average="macro"),
            F1("f1_score_weight", average="weighted"),
        ],
        device=device,
        client_number=args.client_number,
        learning_rate=args.learning_rate,
        heterogeneity_level=args.beta,
        checkpointer=checkpointer,
        freeze_global_feature_extractor=args.freeze_global_extractor,
    )

    fl.client.start_client(server_address=args.server_address, client=client.to_client())
    client.shutdown()