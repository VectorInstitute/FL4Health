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
from fl4health.utils.config import load_config
from fl4health.checkpointing.checkpointer import BestMetricTorchCheckpointer


torch.set_default_dtype(torch.float64)


class SecAggClient(SecureAggregationClient):
    # Supply @abstractmethod implementations

    """
    model, loss, optimizer
    """

    def get_model(self, config: Config) -> Module:
        return Net().to(self.device)

    def get_criterion(self, config: Config) -> _Loss:
        return CrossEntropyLoss()

    def get_optimizer(self, config: Config) -> Optimizer | Dict[str, Optimizer]:
        return SGD(self.model.parameters(), lr=0.001, momentum=0.9)

    """
    training & validation data
    """

    def get_data_loaders(self, config: Config) -> Tuple[DataLoader, DataLoader]:
        batch_size = self.narrow_config_type(config, "batch_size", int)
        # sample_size is currently unused
        training_loader, training_loader = poisson_subsampler_cifar10(self.data_path, batch_size)

        return training_loader, training_loader


if __name__ == "__main__":
    # link to dataset
    parser = ArgumentParser(description="Secure aggregation client.")
    parser.add_argument(
        "--dataset_path",
        action="store",
        type=str,
        default="examples/datasets",
        help="Path to the local dataset, no need to encluse in quotes.",
    )
    parser.add_argument(
        "--config_path",
        action="store",
        type=str,
        help="Path to configuration file. No enclosing quotes required.",
        default="examples/secure_aggregation_example/config.yaml",
    )
    parser.add_argument(
        "--client_number",
        action="store",
        type=int,
        help="Client number. No enclosing quotes required.",
    )
    args = parser.parse_args()
    config = load_config(args.config_path)
    data_path = Path(args.dataset_path)

    # compute resource
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # privacy settings 
    privacy_settings = {
        'clipping_threshold': config['clipping_threshold'],
        'granularity': config['granularity'],
        'noise_scale': config['noise_scale'],
        'bias': config['bias'],
    }

    checkpoint_dir = 'examples/secure_aggregation_example'
    checkpoint_name = f"client_{args.client_number}_best_model.pkl"
    checkpointer = BestMetricTorchCheckpointer(checkpoint_dir, checkpoint_name, maximize=False)

    # instantiate Cifar client class with SecAgg as defined above
    client = SecAggClient(
        data_path=data_path, 
        metrics=[Accuracy("accuracy")], 
        device=DEVICE,
        privacy_settings=privacy_settings,
        task_name='secure_aggregation_example',
        num_mini_clients=8,
        checkpointer=checkpointer,
        client_id=args.client_number

    )

    # NOTE server needs to be started before clients
    RunClient(server_address="0.0.0.0:8080", client=client)
    client.shutdown()
