from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, Tuple

import torch
import flwr as fl
from flwr.common.typing import Config
from torch.nn import CrossEntropyLoss, Module
from torch.nn.modules.loss import _Loss
from torch.optim import SGD, Optimizer, AdamW
from torch.utils.data import DataLoader

from examples.models.cnn_model import Net
from fl4health.clients.ddgm_client import DDGMClient

from logging import DEBUG, INFO, WARN
from flwr.common.logger import log

from fl4health.utils.load_data import load_cifar10_data, load_cifar10_test_data
# , poisson_subsampler_cifar10
from fl4health.utils.metrics import Accuracy
from fl4health.utils.config import load_config
from fl4health.checkpointing.checkpointer import BestLossTorchModuleCheckpointer

from fl4health.utils.config import narrow_dict_type

torch.set_default_dtype(torch.float64)

from fl4health.servers.secure_aggregation_utils import (
    get_model_dimension,
    vectorize_model
)

from fl4health.privacy_mechanisms.slow_discrete_gaussian_mechanism import generate_random_sign_vector, get_exponent

class SecAggClient(DDGMClient):
    # Supply @abstractmethod implementations

    """
    model, loss, optimizer
    """

    def get_model(self, config: Config) -> Module:
        return Net()

    def get_criterion(self, config: Config) -> _Loss:
        return CrossEntropyLoss()

    def get_optimizer(self, config: Config) -> Optimizer:
        # return AdamW(self.model.parameters(), lr=1e-5)
        return SGD(self.model.parameters(), lr=0.001, momentum=0.9)
    
    def set_sign_vector(self) -> None:
        """
        Set the sign vector for the client.
        """
        # Generate a random sign vector
        # len_parameters = len(vectorize_model(self.model))
        len_parameters = get_model_dimension(self.model)
        log(INFO, f"Model dimension: {len_parameters}")
        padded_model_dim = 2**get_exponent(len_parameters)
        log(INFO, f"Padded model dimension: {padded_model_dim}")
        self.sign_vector = generate_random_sign_vector(dim=padded_model_dim, seed=self.privacy_settings["sign_vector_seed"])
        log(INFO, "finished generating sign vector")


    """
    training & validation data
    """
    
    def get_data_loaders(self, config: Config) -> tuple[DataLoader, DataLoader]:
        batch_size = narrow_dict_type(config, "batch_size", int)
        train_loader, val_loader, _ = load_cifar10_data(self.data_path, batch_size)
        return train_loader, val_loader

    def get_test_data_loader(self, config: Config) -> DataLoader | None:
        batch_size = narrow_dict_type(config, "batch_size", int)
        test_loader, _ = load_cifar10_test_data(self.data_path, batch_size)
        return test_loader


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
        default="examples/ddgm_examples/config.yaml",
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
        'noise_multiplier': config['noise_multiplier'],
        'granularity': config['granularity'],
        'clipping_bound': config['clipping_bound'],
        'bias': config['bias'],
        'sign_vector_seed': config['sign_vector_seed'],
    }

    # checkpoint_dir = 'examples/secagg_non_private_example'
    # checkpoint_name = f"client_{args.client_number}_best_model.pkl"
    # checkpointer = BestLossTorchModuleCheckpointer(checkpoint_dir, checkpoint_name)


    # instantiate Cifar client class with SecAgg as defined above
    client = SecAggClient(
        data_path=data_path, 
        metrics=[Accuracy("accuracy")], 
        device=DEVICE,
        privacy_settings=privacy_settings,
    )

    # # NOTE server needs to be started before clients
    # RunClient(server_address="0.0.0.0:8080", client=client)
    # client.shutdown()
    
    fl.client.start_client(server_address="0.0.0.0:8080", client=client.to_client(),insecure=True,)
    client.shutdown()
