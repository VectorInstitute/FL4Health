import argparse
import os

from pathlib import Path
from typing import Optional, Sequence, Tuple

import flwr as fl
import torch
import torch.nn as nn

from flamby.datasets.fed_isic2019 import BATCH_SIZE, LR, NUM_CLIENTS, Baseline, BaselineLoss
from flwr.common.logger import log
from logging import INFO
from flwr.common.typing import Config
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer

from fl4health.checkpointing.checkpointer import BestMetricTorchCheckpointer, TorchCheckpointer
from fl4health.utils.losses import LossMeterType
from fl4health.utils.metrics import BalancedAccuracy, Metric, MetricMeterType
from research.flamby.flamby_data_utils import construct_fedisic_train_val_datasets
from torch.utils.data import DataLoader

from fl4health.utils.config import load_config

from research.isic_custom_models import BaseLineFrozenBN
from fl4health.clients.scaffold_client import DPScaffoldLoggingClient

torch.set_default_device('cuda' if torch.cuda.is_available() else 'cpu')
# torch.set_default_dtype(torch.float64)
from flamby.datasets.fed_isic2019 import Baseline



class FedIsic2019FedAvgClient(DPScaffoldLoggingClient):

    def set_task_name(self):
        self.task_name = 'Fed-ISIC2019 Local'
        return self.task_name
    
    def get_data_loaders(self, config: Config) -> Tuple[DataLoader, DataLoader]:
        train_dataset, validation_dataset = construct_fedisic_train_val_datasets(
            self.client_id, str(self.data_path)
        )
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, generator=torch.Generator(device='cuda' if torch.cuda.is_available() else "cpu"))
        val_loader = DataLoader(validation_dataset, batch_size=8, shuffle=False, generator=torch.Generator(device='cuda' if torch.cuda.is_available() else "cpu"))
        return train_loader, val_loader

    def get_model(self, config: Config) -> nn.Module:
        # model: nn.Module = FedISICImageClassifier().to(self.device)
        return BaseLineFrozenBN().to(self.device)

    def get_optimizer(self, config: Config) -> Optimizer:
        return torch.optim.AdamW(self.model.parameters(), lr=0.001)

    def get_criterion(self, config: Config) -> _Loss:
        return BaselineLoss()


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
        help="Path to the preprocessed FedIsic2019 Dataset (ex. path/to/fedisic2019)",
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
        help="Number of the client for dataset loading (should be 0-5 for FedIsic2019)",
        required=True,
    )
    parser.add_argument(
        "--learning_rate", action="store", type=float, help="Learning rate for local optimization", default=LR
    )
    hyperparameter_options = 'clipping_threshold, granularity, noise_scale, bias, model_integer_range_exponent'
    parser.add_argument(
        "--hyperparameter_name", action="store", type=str, help=f'Tunable hyperparameter type: {hyperparameter_options}.'
    )
    parser.add_argument(
        "--hyperparameter_value", action="store", type=float, help="Tunable hyperparameter value."
    )
    parser.add_argument(
        "--config_path",
        action="store",
        type=str,
        help="Path to configuration file.",
        default="config.yaml",
    )
    args = parser.parse_args()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(INFO, f"Device to be used: {DEVICE}")
    log(INFO, f"Server Address: {args.server_address}")
    log(INFO, f"Learning Rate: {args.learning_rate}")

    checkpoint_dir = os.path.join(args.artifact_dir, args.run_name)
    checkpoint_name = f"client_{args.client_number}_best_model.pkl"
    checkpointer = BestMetricTorchCheckpointer(checkpoint_dir, checkpoint_name, maximize=False)

    config = load_config(args.config_path)
    
    client = FedIsic2019FedAvgClient(
        data_path=Path(args.dataset_dir),
        metrics=[BalancedAccuracy("FedIsic2019_balanced_accuracy")],
        device=DEVICE,
        client_id=args.client_number,
        checkpointer=checkpointer,
        metric_meter_type= MetricMeterType.ACCUMULATION
    )

    fl.client.start_numpy_client(server_address=args.server_address, client=client, grpc_max_message_length=1600000000,)

    # Shutdown the client gracefully
    client.shutdown()
