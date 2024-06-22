import argparse
from logging import INFO
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import flwr as fl
import torch
from flwr.common.logger import log
from flwr.common.typing import Config, Tuple
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from examples.models.parallel_split_cnn import ParallelSplitHeadClassifier
from examples.models.sequential_split_models import (
    SequentialGlobalFeatureExtractorMnist,
    SequentialLocalPredictionHeadMnist,
)
from fl4health.clients.fenda_ditto_client import FendaDittoClient
from fl4health.model_bases.fenda_base import FendaModel
from fl4health.model_bases.parallel_split_models import ParallelFeatureJoinMode
from fl4health.model_bases.sequential_split_models import SequentiallySplitExchangeBaseModel
from fl4health.checkpointing.checkpointer import BestLossTorchCheckpointer
from fl4health.checkpointing.client_module import ClientCheckpointModule
from fl4health.utils.losses import LossMeterType
from fl4health.utils.load_data import load_mnist_data
from fl4health.utils.metrics import Accuracy, Metric
from fl4health.utils.random import set_all_random_seeds
from fl4health.utils.sampler import DirichletLabelBasedSampler


class MnistFendaDittoClient(FendaDittoClient):
    def __init__(
        self,
        data_path: Path,
        metrics: Sequence[Metric],
        device: torch.device,
        checkpoint_dir: str,
        loss_meter_type: LossMeterType = LossMeterType.AVERAGE,
        checkpointer: Optional[ClientCheckpointModule] = None,
        lam: float = 1.0,
        freeze_global_feature_extractor: bool = False,
    ) -> None:
        super().__init__(
            data_path=data_path,
            metrics=metrics,
            device=device,
            loss_meter_type=loss_meter_type,
            checkpointer=checkpointer,
            lam=lam,
            freeze_global_feature_extractor=freeze_global_feature_extractor,
        )

        pre_aggregation_checkpointer = BestLossTorchCheckpointer(checkpoint_dir, f"client_{self.client_name}_pre_agg.pkl")
        post_aggregation_checkpointer = BestLossTorchCheckpointer(checkpoint_dir, f"client_{self.client_name}_post_agg.pkl")
        self.checkpointer = ClientCheckpointModule(
            pre_aggregation=pre_aggregation_checkpointer, post_aggregation=post_aggregation_checkpointer
        )

    def get_data_loaders(self, config: Config) -> Tuple[DataLoader, DataLoader]:
        sampler = DirichletLabelBasedSampler(list(range(10)), sample_percentage=0.75, beta=1)
        batch_size = self.narrow_config_type(config, "batch_size", int)
        train_loader, val_loader, _ = load_mnist_data(self.data_path, batch_size, sampler)
        return train_loader, val_loader

    def get_global_model(self, config: Config) -> SequentiallySplitExchangeBaseModel:
        return SequentiallySplitExchangeBaseModel(
            base_module=SequentialGlobalFeatureExtractorMnist(),
            head_module=SequentialLocalPredictionHeadMnist(),
        ).to(self.device)

    def get_model(self, config: Config) -> FendaModel:
        return FendaModel(
            SequentialGlobalFeatureExtractorMnist(),
            SequentialGlobalFeatureExtractorMnist(),
            ParallelSplitHeadClassifier(ParallelFeatureJoinMode.CONCATENATE),
        ).to(self.device)

    def get_optimizer(self, config: Config) -> Dict[str, Optimizer]:
        # Note that the global optimizer operates on self.global_model.parameters()
        global_optimizer = torch.optim.AdamW(self.global_model.parameters(), lr=0.01)
        local_optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.01)
        return {"global": global_optimizer, "local": local_optimizer}

    def get_criterion(self, config: Config) -> _Loss:
        return torch.nn.CrossEntropyLoss()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FL Client Main")
    parser.add_argument("--dataset_path", action="store", type=str, help="Path to the local dataset")
    parser.add_argument(
        "--server_address",
        action="store",
        type=str,
        help="Server Address for the clients to communicate with the server through",
        default="0.0.0.0:8080",
    )
    parser.add_argument(
        "--seed",
        action="store",
        type=int,
        help="Seed for the random number generators across python, torch, and numpy",
        required=False,
    )
    parser.add_argument(
        "--checkpoint_path",
        action="store",
        type=str,
        help="Path to the directory where the checkpoints are stored",
        required=True,
    )
    args = parser.parse_args()

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_path = Path(args.dataset_path)
    log(INFO, f"Device to be used: {DEVICE}")
    log(INFO, f"Server Address: {args.server_address}")

    # Set the random seed for reproducibility
    set_all_random_seeds(args.seed)

    client = MnistFendaDittoClient(data_path, [Accuracy()], DEVICE, lam=0.1)
    fl.client.start_client(server_address=args.server_address, client=client.to_client())

    # Shutdown the client gracefully
    client.shutdown()

    client.metrics_reporter.dump()
