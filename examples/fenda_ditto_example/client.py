import argparse
from logging import INFO
from pathlib import Path

import flwr as fl
import torch
from flwr.common.logger import log
from flwr.common.typing import Config
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from examples.models.parallel_split_cnn import ParallelSplitHeadClassifier
from examples.models.sequential_split_models import (
    SequentialGlobalFeatureExtractorMnist,
    SequentialLocalPredictionHeadMnist,
)
from fl4health.checkpointing.checkpointer import BestLossTorchModuleCheckpointer
from fl4health.checkpointing.client_module import ClientCheckpointAndStateModule
from fl4health.clients.fenda_ditto_client import FendaDittoClient
from fl4health.metrics import Accuracy
from fl4health.model_bases.fenda_base import FendaModel
from fl4health.model_bases.parallel_split_models import ParallelFeatureJoinMode
from fl4health.model_bases.sequential_split_models import SequentiallySplitModel
from fl4health.reporting import JsonReporter
from fl4health.utils.config import narrow_dict_type
from fl4health.utils.load_data import load_mnist_data
from fl4health.utils.random import set_all_random_seeds
from fl4health.utils.sampler import DirichletLabelBasedSampler


class MnistFendaDittoClient(FendaDittoClient):
    def get_data_loaders(self, config: Config) -> tuple[DataLoader, DataLoader]:
        sampler = DirichletLabelBasedSampler(list(range(10)), sample_percentage=0.75, beta=1)
        batch_size = narrow_dict_type(config, "batch_size", int)
        train_loader, val_loader, _ = load_mnist_data(self.data_path, batch_size, sampler)
        return train_loader, val_loader

    def get_global_model(self, config: Config) -> SequentiallySplitModel:
        return SequentiallySplitModel(
            base_module=SequentialGlobalFeatureExtractorMnist(),
            head_module=SequentialLocalPredictionHeadMnist(),
        ).to(self.device)

    def get_model(self, config: Config) -> FendaModel:
        return FendaModel(
            SequentialGlobalFeatureExtractorMnist(),
            SequentialGlobalFeatureExtractorMnist(),
            ParallelSplitHeadClassifier(ParallelFeatureJoinMode.CONCATENATE),
        ).to(self.device)

    def get_optimizer(self, config: Config) -> dict[str, Optimizer]:
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
        required=False,
        default="examples/fenda_ditto_example/",
    )
    parser.add_argument(
        "--checkpointer_type",
        action="store",
        type=str,
        choices=["pre", "post", "both"],
        help="Type of checkpointer to use: pre, post, or both",
        default="both",
    )
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_path = Path(args.dataset_path)
    log(INFO, f"Device to be used: {device}")
    log(INFO, f"Server Address: {args.server_address}")

    # Set the random seed for reproducibility
    set_all_random_seeds(args.seed)

    pre_aggregation_checkpointer = None
    post_aggregation_checkpointer = None

    if args.checkpointer_type in ["pre", "both"]:
        pre_aggregation_checkpointer = BestLossTorchModuleCheckpointer(
            args.checkpoint_path, "fenda_ditto_client_pre_agg.pkl"
        )
    if args.checkpointer_type in ["post", "both"]:
        post_aggregation_checkpointer = BestLossTorchModuleCheckpointer(
            args.checkpoint_path, "fenda_ditto_client_post_agg.pkl"
        )

    checkpoint_and_state_module = ClientCheckpointAndStateModule(
        pre_aggregation=pre_aggregation_checkpointer,
        post_aggregation=post_aggregation_checkpointer,
    )
    client = MnistFendaDittoClient(
        data_path,
        [Accuracy()],
        device,
        args.checkpoint_path,
        checkpoint_and_state_module=checkpoint_and_state_module,
        reporters=[JsonReporter()],
    )
    fl.client.start_client(server_address=args.server_address, client=client.to_client())

    # Shutdown the client gracefully
    client.shutdown()
