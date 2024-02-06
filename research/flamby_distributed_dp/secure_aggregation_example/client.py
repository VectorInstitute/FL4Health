from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, Tuple

import torch
from flwr.client import start_numpy_client as RunClient
from flwr.common.typing import Config
from torch.nn import CrossEntropyLoss, Module
from torch.nn.modules.loss import _Loss
from torch.optim import SGD, Optimizer
from torch.utils.data import DataLoader

from examples.models.cnn_model import Net
from fl4health.clients.secure_aggregation_client import SecureAggregationClient
from fl4health.utils.load_data import load_cifar10_data, poisson_subsampler_cifar10
from fl4health.utils.metrics import Accuracy

from flamby.datasets.fed_heart_disease import BATCH_SIZE, LR, NUM_CLIENTS, Baseline, BaselineLoss
from research.flamby.flamby_data_utils import construct_fed_heard_disease_train_val_datasets


torch.set_default_dtype(torch.float64)


class SecAggClient(SecureAggregationClient):
    # Supply @abstractmethod implementations

    """
    model, loss, optimizer
    """

    def get_model(self, config: Config) -> Module:
        return Baseline().to(self.device)

    def get_criterion(self, config: Config) -> _Loss:
        return BaselineLoss()

    def get_optimizer(self, config: Config) -> Optimizer | Dict[str, Optimizer]:
        return torch.optim.AdamW(self.model.parameters(), lr=0.001)

    """
    training & validation data
    """

    def get_data_loaders(self, config: Config) -> Tuple[DataLoader, DataLoader]:
        batch_size = self.narrow_config_type(config, "batch_size", int)
        
        train_dataset, validation_dataset = construct_fed_heard_disease_train_val_datasets(
            dataset_dir='flamby_datasets/fed_heart_disease',
            pooled=True,
            client_number=0
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator(device='cuda'))
        val_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, generator=torch.Generator(device='cuda'))
        return train_loader, val_loader


if __name__ == "__main__":
    # link to dataset
    parser = ArgumentParser(description="Secure aggregation FedHeartDisease client.")
    parser.add_argument(
        "--dataset_path",
        action="store",
        type=str,
        default="flamby_datasets/fed_heart_disease",
        help="Path to the local dataset, no need to encluse in quotes.",
    )
    args = parser.parse_args()
    data_path = Path(args.dataset_path)

    # compute resource
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # instantiate Cifar client class with SecAgg as defined above
    client = SecAggClient(data_path, [Accuracy("accuracy")], DEVICE)

    # NOTE server needs to be started before clients
    RunClient(server_address="0.0.0.0:8080", client=client)
    client.shutdown()
