import argparse
from pathlib import Path
from logging import INFO
from typing import Dict, Optional, Sequence, Tuple
import os 
import flwr as fl
import torch
import torch.nn as nn
from flwr.common.typing import Config
from flwr.common.logger import log


from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from examples.models.cnn_model import MnistNet

from fl4health.clients.scaffold_client import DPScaffoldLoggingClient
from fl4health.utils.load_data import load_mnist_data
from fl4health.utils.metrics import Accuracy
from fl4health.utils.sampler import DirichletLabelBasedSampler
from fl4health.checkpointing.checkpointer import BestMetricTorchCheckpointer, TorchCheckpointer
from fl4health.utils.losses import Losses, LossMeterType
from fl4health.utils.metrics import Metric, MetricMeterType

from flamby.datasets.fed_heart_disease import BATCH_SIZE, LR, NUM_CLIENTS, Baseline, BaselineLoss
from research.flamby.flamby_data_utils import construct_fed_heard_disease_train_val_datasets
from research.flamby.fed_heart_disease.large_baseline import FedHeartDiseaseLargeBaseline


class FedHeartClient(DPScaffoldLoggingClient):

    def set_task_name(self):
        self.task_name = "Fed-HeartDisease Local"
        return self.task_name

    def get_data_loaders(self, config: Config) -> Tuple[DataLoader, DataLoader]:
        train_dataset, validation_dataset = construct_fed_heard_disease_train_val_datasets(
            self.client_id, str(self.data_path)
        )
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, generator=torch.Generator(device='cuda' if torch.cuda.is_available() else "cpu"))
        val_loader = DataLoader(validation_dataset, batch_size=64, shuffle=False, generator=torch.Generator(device='cuda' if torch.cuda.is_available() else "cpu"))
        return train_loader, val_loader

    def get_optimizer(self, config: Config) -> Optimizer:
        return torch.optim.SGD(self.model.parameters(), lr=0.05)

    def get_model(self, config: Config) -> nn.Module:
        return FedHeartDiseaseLargeBaseline().to(self.device)

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
        help="Path to the preprocessed Fed Heart Disease Dataset (ex. path/to/fed_heart_disease)",
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
        help="Number of the client for dataset loading (should be 0-3 for Fed Heart Disease)",
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

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_path = Path(args.dataset_dir)



    checkpoint_dir = os.path.join(args.artifact_dir, args.run_name)
    checkpoint_name = f"client_{args.client_number}_best_model.pkl"
    checkpointer = BestMetricTorchCheckpointer(checkpoint_dir, checkpoint_name, maximize=False)


    client = FedHeartClient(data_path=data_path, 
                            metrics=[Accuracy('FedHeartDisease_accuracy')], 
                            device=DEVICE,
                              checkpointer=checkpointer, 
                              client_id=args.client_number)

    fl.client.start_numpy_client(server_address=args.server_address, client=client)
    client.shutdown()
