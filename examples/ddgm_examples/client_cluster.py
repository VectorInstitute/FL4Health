from argparse import ArgumentParser
from pathlib import Path

import torch
import flwr as fl
from flwr.common.typing import Config
from torch.nn import CrossEntropyLoss, Module
from torch.nn.modules.loss import _Loss
from torch.optim import SGD, Optimizer
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from examples.models.cnn_model import Net, MnistNet
from fl4health.clients.ddgm_client import DDGMClient

from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import NaturalIdPartitioner

from logging import INFO
from flwr.common.logger import log

from fl4health.utils.load_data import load_mnist_data

from fl4health.utils.metrics import Accuracy
from fl4health.utils.config import load_config

from fl4health.utils.config import narrow_dict_type

torch.set_default_dtype(torch.float64)

from fl4health.servers.secure_aggregation_utils import (
    get_model_dimension
)

from fl4health.privacy_mechanisms.slow_discrete_gaussian_mechanism import generate_random_sign_vector, get_exponent

class SecAggClient(DDGMClient):
    """
    model, loss, optimizer
    """

    def get_model(self, config: Config) -> Module:
        return MnistNet().to(self.device)

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
        len_parameters = get_model_dimension(self.model)
        log(INFO, f"Model dimension: {len_parameters}")
        padded_model_dim = 2**get_exponent(len_parameters)
        log(INFO, f"Padded model dimension: {padded_model_dim}")
        self.sign_vector = generate_random_sign_vector(dim=padded_model_dim, seed=self.privacy_settings["sign_vector_seed"]).to(self.device)
        log(INFO, "finished generating sign vector")


    """
    training & validation data
    """
    def get_data_loaders(self, config: Config) -> tuple[DataLoader, DataLoader]:

        if self.dataset_name == "mnist":
            log(INFO, f'loading {self.dataset_name} data')
            batch_size = narrow_dict_type(config, "batch_size", int)
            train_loader, val_loader, _ = load_mnist_data(self.data_path, batch_size)
            self.dataset_size = len(train_loader.dataset)
            log(INFO, f'size of data: {self.dataset_size}')

            return train_loader, val_loader
        elif self.dataset_name == "femnist":
            log(INFO, f'loading {self.dataset_name} data')
            fds = FederatedDataset(
                dataset="flwrlabs/femnist",
                partitioners={"train": NaturalIdPartitioner(partition_by="writer_id")}
            )
            partition = fds.load_partition(partition_id=self.client_number)

            # split into train and validation sets
            split_dict = partition.train_test_split(test_size = 0.2)
            train, val = split_dict['train'], split_dict['test']

            transforms = ToTensor()
            train_torch = train.map(
                lambda img: {"image": transforms(img)}, input_columns="image"
            ).with_format("torch")
            val_torch = val.map(
                lambda img: {"image": transforms(img)}, input_columns="image"
            ).with_format("torch")

            train_loader = DataLoader(train_torch, batch_size=20)
            val_loader = DataLoader(val_torch, batch_size=20)
            pass
        else:
            raise NotImplementedError


if __name__ == "__main__":
    # link to dataset
    parser = ArgumentParser(description="Secure aggregation client.")
    parser.add_argument(
        "--dataset_path",
        action="store",
        type=str,
        default="examples/datasets/mnist_data/",
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
        required=True,
        help="Client number.",
    )
    args = parser.parse_args()
    config = load_config(args.config_path)
    data_path = Path(args.dataset_path)

    # compute resource
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # privacy settings 
    privacy_settings = {
        'enable_dp': config['enable_dp'],
        'noise_multiplier': config['noise_multiplier'],
        'granularity': config['granularity'],
        'clipping_bound': config['clipping_bound'],
        'bias': config['bias'],
        'sign_vector_seed': config['sign_vector_seed'],
        'dataset': config['dataset']
    }

    # instantiate Cifar client class with SecAgg as defined above
    client = SecAggClient(
        data_path=data_path, 
        metrics=[Accuracy("accuracy")], 
        device=device,
        privacy_settings=privacy_settings,
        client_number=args.client_number
    )
    
    fl.client.start_client(server_address="0.0.0.0:8081", client=client.to_client())
    client.shutdown()
