from argparse import ArgumentParser
from pathlib import Path
import os

import torch
import flwr as fl
from flwr.common.typing import Config
from torch.nn import CrossEntropyLoss, Module
from torch.nn.modules.loss import _Loss
from torch.optim import SGD, Optimizer, Adam
from torch.utils.data import DataLoader

from examples.models.cnn_model import Net, MnistNet, FEMnistNet
from fl4health.clients.basic_client import BasicClient

from logging import INFO
from flwr.common.logger import log

from fl4health.utils.load_data import load_mnist_data, load_femnist_data
from fl4health.utils.metrics import Accuracy
from fl4health.utils.config import load_config
from fl4health.utils.config import narrow_dict_type

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

class FemnistClient(BasicClient):
    def __init__(self, *args, client_number:int=None, **kwargs):
        self.client_number = client_number
        super().__init__(*args, **kwargs)

    def get_model(self, config: Config) -> Module:
        return FEMnistNet().to(self.device)
        
    def get_criterion(self, config: Config) -> _Loss:
        return CrossEntropyLoss()

    def get_optimizer(self, config: Config) -> Optimizer:
        return Adam(self.model.parameters(), lr=0.001)
        # return SGD(self.model.parameters(), lr=0.001, momentum=0.9)

    def get_data_loaders(self, config: Config) -> tuple[DataLoader, DataLoader]:
        log(INFO, f'loading FEMNIST data for client partition {self.client_number}')
        batch_size = narrow_dict_type(config, "batch_size", int)

        train_loader, val_loader, _ = load_femnist_data(batch_size, self.client_number, partition_by="writer_id", num_workers=0)

        self.dataset_size = len(train_loader.dataset)
        log(INFO, f'size of data: {self.dataset_size}')
        
        return train_loader, val_loader

if __name__ == "__main__":
    # link to dataset
    parser = ArgumentParser(description="FL client with FEMNIST Data.")
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
        default="examples/basic_femnist_example/config.yaml",
    )
    parser.add_argument(
        "--client_number",
        action="store",
        type=int,
        required=True,
        help="Client number.",
    )
    parser.add_argument(
        "--server_address",
        action="store",
        type=str,
        default="0.0.0.0:8081",
    )
    args = parser.parse_args()
    config = load_config(args.config_path)
    data_path = Path(args.dataset_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    client = FemnistClient(
        data_path=data_path, 
        metrics=[Accuracy("accuracy")], 
        device=device,
        client_number=args.client_number
    )
    
    fl.client.start_client(server_address=args.server_address, client=client.to_client())
    client.shutdown()
