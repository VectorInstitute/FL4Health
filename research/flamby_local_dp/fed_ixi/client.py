import argparse
import os

from pathlib import Path
from typing import Optional, Sequence, Tuple

import flwr as fl
import torch
import torch.nn as nn

from flamby.datasets.fed_ixi import BATCH_SIZE, LR, NUM_CLIENTS, Baseline, BaselineLoss
from flwr.common.logger import log
from logging import INFO
from flwr.common.typing import Config
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from fl4health.utils.metrics import BinarySoftDiceCoefficient, Metric, MetricMeterType

from fl4health.checkpointing.checkpointer import BestMetricTorchCheckpointer, TorchCheckpointer, BestMetricCheckpointWeights
from fl4health.utils.losses import LossMeterType
from fl4health.utils.metrics import BalancedAccuracy, Metric, MetricMeterType
from research.flamby.flamby_data_utils import construct_fed_ixi_train_val_datasets
from torch.utils.data import DataLoader

from fl4health.utils.config import load_config

from research.flamby_local_dp.fed_ixi.model import ModifiedBaseline, FedIXIUNet

from fl4health.clients.scaffold_client import DPScaffoldLoggingClient

torch.set_default_device('cuda' if torch.cuda.is_available() else 'cpu')
# torch.set_default_dtype(torch.float64)
from flamby.datasets.fed_ixi import Baseline

class FedIXIFedAvgClient(DPScaffoldLoggingClient):

    def set_task_name(self):
        self.task_name ='Fed-IXI Local'
        return self.task_name
    
    def get_data_loaders(self, config: Config) -> Tuple[DataLoader, DataLoader]:
        train_dataset, validation_dataset = construct_fed_ixi_train_val_datasets(
            self.client_id, str(self.data_path)
        )
        train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, generator=torch.Generator(device='cuda' if torch.cuda.is_available() else "cpu"))
        val_loader = DataLoader(validation_dataset, batch_size=2, shuffle=False, generator=torch.Generator(device='cuda' if torch.cuda.is_available() else "cpu"))
        return train_loader, val_loader

    def get_model(self, config: Config) -> nn.Module:
        model: nn.Module = FedIXIUNet().to(self.device)
        return model

    def get_optimizer(self, config: Config) -> Optimizer:
        # return torch.optim.AdamW(self.model.parameters(), lr=0.05)
        return torch.optim.SGD(self.model.parameters(), lr=0.05)

    def get_criterion(self, config: Config) -> _Loss:
        return BaselineLoss()

# if __name__ == '__main__':
#     test_ixi_batch = torch.rand(1,1,48,60,48)
#     model = FedIXIUNet()
#     optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
#     loss = BaselineLoss()

#     from opacus import PrivacyEngine
#     privacy_engine = PrivacyEngine()
#     train_dataset, validation_dataset = construct_fed_ixi_train_val_datasets(
#             0, str('flamby_datasets/fed_ixi')
#     )
#     train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, generator=torch.Generator(device='cuda' if torch.cuda.is_available() else "cpu"))
#     model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
#     module=model,
#     optimizer=optimizer,
#     data_loader=train_loader,
#     epochs=1,
#     target_epsilon=10,
#     target_delta=10,
#     max_grad_norm=5,
#     )

#     print('here1')

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
    checkpointer = BestMetricCheckpointWeights(checkpoint_dir, checkpoint_name, maximize=False)

    config = load_config(args.config_path)

    
    client = FedIXIFedAvgClient(
        data_path=Path(args.dataset_dir),
        metrics=[BinarySoftDiceCoefficient("FedIXI_dice")],
        device=DEVICE,
        client_id=args.client_number,
        checkpointer=checkpointer,
    )

    fl.client.start_numpy_client(server_address=args.server_address, client=client)

    # Shutdown the client gracefully
    client.shutdown()
