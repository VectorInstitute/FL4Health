import argparse
from pathlib import Path

import flwr as fl
import torch
from flwr.common.typing import Config
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from examples.models.cnn_model import Net
from fl4health.clients.partial_weight_exchange_client import PartialWeightExchangeClient
from fl4health.metrics import Accuracy
from fl4health.parameter_exchange.parameter_exchanger_base import ParameterExchanger
from fl4health.parameter_exchange.parameter_selection_criteria import largest_final_magnitude_scores
from fl4health.parameter_exchange.sparse_coo_parameter_exchanger import SparseCooParameterExchanger
from fl4health.utils.config import narrow_dict_type
from fl4health.utils.load_data import load_cifar10_data


class CifarSparseCooTensorClient(PartialWeightExchangeClient):
    def get_data_loaders(self, config: Config) -> tuple[DataLoader, DataLoader]:
        batch_size = narrow_dict_type(config, "batch_size", int)
        train_loader, val_loader, _ = load_cifar10_data(self.data_path, batch_size)
        return train_loader, val_loader

    def get_criterion(self, config: Config) -> _Loss:
        return torch.nn.CrossEntropyLoss()

    def get_optimizer(self, config: Config) -> Optimizer:
        return torch.optim.AdamW(self.model.parameters(), lr=0.001)

    def get_model(self, config: Config) -> nn.Module:
        return Net().to(self.device)

    def get_parameter_exchanger(self, config: Config) -> ParameterExchanger:
        sparsity_level = narrow_dict_type(config, "sparsity_level", float)
        # The user may pass in a different score_gen_function to allow for alternative
        # selection criterion.
        return SparseCooParameterExchanger(
            sparsity_level=sparsity_level,
            score_gen_function=largest_final_magnitude_scores,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FL Client Main")
    parser.add_argument("--dataset_path", action="store", type=str, help="Path to the local dataset")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_path = Path(args.dataset_path)
    client = CifarSparseCooTensorClient(data_path, [Accuracy("accuracy")], device)
    fl.client.start_client(server_address="0.0.0.0:8080", client=client.to_client())
    client.shutdown()
