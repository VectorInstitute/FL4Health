import argparse
from pathlib import Path
from typing import Tuple

import flwr as fl
import torch
import torch.nn as nn
from flwr.common.typing import Config
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from examples.models.cnn_model import Net
from fl4health.clients.partial_weight_exchange_client import PartialWeightExchangeClient
from fl4health.parameter_exchange.parameter_exchanger_base import ParameterExchanger
from fl4health.parameter_exchange.parameter_selection_criteria import largest_final_magnitude_scores
from fl4health.parameter_exchange.sparse_coo_parameter_exchanger import SparseCooParameterExchanger
from fl4health.utils.load_data import load_cifar10_data
from fl4health.utils.metrics import Accuracy


class CifarSparseCooTensorClient(PartialWeightExchangeClient):
    def get_data_loaders(self, config: Config) -> Tuple[DataLoader, DataLoader]:
        batch_size = self.narrow_config_type(config, "batch_size", int)
        train_loader, val_loader, _ = load_cifar10_data(self.data_path, batch_size)
        return train_loader, val_loader

    def get_criterion(self, config: Config) -> _Loss:
        return torch.nn.CrossEntropyLoss()

    def get_optimizer(self, config: Config) -> Optimizer:
        return torch.optim.AdamW(self.model.parameters(), lr=0.001)

    def get_model(self, config: Config) -> nn.Module:
        return Net().to(self.device)

    def get_parameter_exchanger(self, config: Config) -> ParameterExchanger:
        sparsity_level = self.narrow_config_type(config, "sparsity_level", float)
        # The user may pass in a different score_gen_function to allow for alternative
        # selection criterion.
        parameter_exchanger = SparseCooParameterExchanger(
            sparsity_level=sparsity_level,
            score_gen_function=largest_final_magnitude_scores,
        )
        return parameter_exchanger


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FL Client Main")
    parser.add_argument("--dataset_path", action="store", type=str, help="Path to the local dataset")
    args = parser.parse_args()

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_path = Path(args.dataset_path)
    client = CifarSparseCooTensorClient(data_path, [Accuracy("accuracy")], DEVICE)
    fl.client.start_client(server_address="0.0.0.0:8080", client=client.to_client())
    client.shutdown()
