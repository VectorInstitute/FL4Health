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
from fl4health.parameter_exchange.layer_exchanger import DynamicLayerExchanger
from fl4health.parameter_exchange.parameter_exchanger_base import ParameterExchanger
from fl4health.parameter_exchange.parameter_selection_criteria import LayerSelectionFunctionConstructor
from fl4health.utils.config import narrow_dict_type
from fl4health.utils.load_data import load_cifar10_data
from fl4health.utils.sampler import DirichletLabelBasedSampler


class CifarDynamicLayerClient(PartialWeightExchangeClient):
    def get_data_loaders(self, config: Config) -> tuple[DataLoader, DataLoader]:
        batch_size = narrow_dict_type(config, "batch_size", int)
        sample_percentage = narrow_dict_type(config, "sample_percentage", float)
        beta = narrow_dict_type(config, "beta", float)
        assert beta > 0 and 0 < sample_percentage < 1
        sampler = DirichletLabelBasedSampler(list(range(10)), sample_percentage=sample_percentage, beta=beta)
        train_loader, val_loader, _ = load_cifar10_data(self.data_path, batch_size, sampler=sampler)
        return train_loader, val_loader

    def get_criterion(self, config: Config) -> _Loss:
        return torch.nn.CrossEntropyLoss()

    def get_optimizer(self, config: Config) -> Optimizer:
        return torch.optim.AdamW(self.model.parameters(), lr=0.001)

    def get_model(self, config: Config) -> nn.Module:
        return Net().to(self.device)

    def get_parameter_exchanger(self, config: Config) -> ParameterExchanger:
        norm_threshold = narrow_dict_type(config, "norm_threshold", float)
        exchange_percentage = narrow_dict_type(config, "exchange_percentage", float)
        normalize = narrow_dict_type(config, "normalize", bool)
        select_drift_more = narrow_dict_type(config, "select_drift_more", bool)
        filter_by_percentage = narrow_dict_type(config, "filter_by_percentage", bool)

        layer_selection_function_constructor = LayerSelectionFunctionConstructor(
            norm_threshold, exchange_percentage, normalize, select_drift_more
        )

        layer_selection_function = (
            layer_selection_function_constructor.select_by_percentage()
            if filter_by_percentage
            else layer_selection_function_constructor.select_by_threshold()
        )

        return DynamicLayerExchanger(layer_selection_function=layer_selection_function)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FL Client Main")
    parser.add_argument("--dataset_path", action="store", type=str, help="Path to the local dataset")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_path = Path(args.dataset_path)
    client = CifarDynamicLayerClient(data_path, [Accuracy("accuracy")], device, store_initial_model=True)
    fl.client.start_client(server_address="0.0.0.0:8080", client=client.to_client())
    client.shutdown()
