import argparse
import os
from logging import INFO
from pathlib import Path
from typing import Optional, Sequence, Tuple

import flwr as fl
import torch
import torch.nn as nn
from flamby.datasets.fed_ixi import BATCH_SIZE, LR, NUM_CLIENTS, Baseline, BaselineLoss
from flwr.common.logger import log
from flwr.common.typing import Config
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from fl4health.checkpointing.checkpointer import BestMetricTorchCheckpointer, TorchCheckpointer
from fl4health.clients.basic_client import BasicClient
from fl4health.utils.losses import LossMeterType
from fl4health.utils.metrics import BinarySoftDiceCoefficient, Metric, MetricMeterType
from research.flamby.flamby_data_utils import construct_fed_ixi_train_val_datasets


from fl4health.utils.config import load_config
from research.flamby_central_dp.fed_ixi.model import ModifiedBaseline

from fl4health.clients.central_dp_client import CentralDPClient
torch.set_default_device('cuda' if torch.cuda.is_available() else 'cpu')

import torch
from research.flamby_local_dp.fed_ixi.model import ModifiedBaseline, FedIXIUNet


class FedIxiFedAvgClient(CentralDPClient):
    def __init__(
        self,
        data_path: Path,
        metrics: Sequence[Metric],
        device: torch.device,
        client_number: int,
        learning_rate: float,
        loss_meter_type: LossMeterType = LossMeterType.AVERAGE,
        metric_meter_type: MetricMeterType = MetricMeterType.ACCUMULATION,
        checkpointer: Optional[TorchCheckpointer] = None,
    ) -> None:
        super().__init__(
            data_path=data_path,
            metrics=metrics,
            device=device,
            loss_meter_type=loss_meter_type,
            metric_meter_type=metric_meter_type,
            checkpointer=checkpointer,
            client_id=client_number,
        )
        self.client_number = client_number
        self.learning_rate = learning_rate

        assert 0 <= client_number < NUM_CLIENTS
        log(INFO, f"Client Name: {self.client_name}, Client Number: {self.client_number}")

    def get_data_loaders(self, config: Config) -> Tuple[DataLoader, DataLoader]:
        train_dataset, validation_dataset = construct_fed_ixi_train_val_datasets(
            self.client_number, str(self.data_path)
        )
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  generator=torch.Generator(device='cuda' if torch.cuda.is_available() else "cpu"))
        val_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False,  generator=torch.Generator(device='cuda' if torch.cuda.is_available() else "cpu"))
        return train_loader, val_loader

    def get_model(self, config: Config) -> nn.Module:
        # NOTE: We set the out_channels_first_layer to 12 rather than the default of 8. This roughly doubles the size
        # of the baseline model to be used (1106520 DOF). This is to allow for a fair parameter comparison with FENDA
        # and APFL
        model: nn.Module = FedIXIUNet().to(self.device)
        return model

    def get_optimizer(self, config: Config) -> Optimizer:
        return torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)

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
        help="Path to the preprocessed FedIXI Dataset (ex. path/to/fedixi/dataset)",
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
        help="Number of the client for dataset loading (should be 0-2 for FedIXI)",
        required=True,
    )
    parser.add_argument(
        "--config_path",
        action="store",
        type=str,
        help="Path to configuration file.",
        default="config.yaml",
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
    args = parser.parse_args()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(INFO, f"Device to be used: {DEVICE}")
    log(INFO, f"Server Address: {args.server_address}")
    log(INFO, f"Learning Rate: {args.learning_rate}")

    checkpoint_dir = os.path.join(args.artifact_dir, args.run_name)
    checkpoint_name = f"client_{args.client_number}_best_model.pkl"
    checkpointer = BestMetricTorchCheckpointer(checkpoint_dir, checkpoint_name, maximize=False)

    config = load_config(args.config_path)
    privacy_settings = {
        'gaussian_noise_variance': config['gaussian_noise_variance'],
    }

    # update privacy setting for tunable hyperparameter
    key, value = args.hyperparameter_name, args.hyperparameter_value
    assert key in ['gaussian_noise_variance']
    log(INFO, f'{type(key)}, {key}, {type(value)}, {value}')
    privacy_settings[key] = value
    log(INFO, f'{privacy_settings}')

    client = FedIxiFedAvgClient(
        data_path=Path(args.dataset_dir),
        metrics=[BinarySoftDiceCoefficient("FedIXI_dice")],
        device=DEVICE,
        client_number=args.client_number,
        learning_rate=args.learning_rate,
        checkpointer=checkpointer,
    )

    fl.client.start_numpy_client(server_address=args.server_address, client=client)

    # Shutdown the client gracefully
    client.shutdown()