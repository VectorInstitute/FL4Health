from pathlib import Path

import pytest
import torch
from flwr.common.typing import Config
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset

from fl4health.clients.fedbn_client import FedBnClient
from fl4health.metrics import Accuracy
from fl4health.parameter_exchange.full_exchanger import FullParameterExchanger
from fl4health.parameter_exchange.layer_exchanger import LayerExchangerWithExclusions
from fl4health.parameter_exchange.parameter_exchanger_base import ParameterExchanger
from tests.test_utils.models_for_test import ToyConvNet


class GoodClientForTest(FedBnClient):
    def get_parameter_exchanger(self, config: Config) -> ParameterExchanger:
        return LayerExchangerWithExclusions(ToyConvNet(include_bn=True), {nn.BatchNorm1d})

    def get_data_loaders(self, config: Config) -> tuple[DataLoader, DataLoader]:
        train_loader = DataLoader(TensorDataset(torch.ones((4, 4)), torch.ones((4))))
        val_loader = DataLoader(TensorDataset(torch.ones((4, 4)), torch.ones((4))))
        return train_loader, val_loader

    def get_criterion(self, config: Config) -> _Loss:
        return torch.nn.CrossEntropyLoss()

    def get_optimizer(self, config: Config) -> Optimizer:
        return torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

    def get_model(self, config: Config) -> nn.Module:
        return ToyConvNet().to(self.device)


class BadClientForTest(FedBnClient):
    def get_parameter_exchanger(self, config: Config) -> ParameterExchanger:
        return FullParameterExchanger()

    def get_data_loaders(self, config: Config) -> tuple[DataLoader, DataLoader]:
        train_loader = DataLoader(TensorDataset(torch.ones((4, 4)), torch.ones((4))))
        val_loader = DataLoader(TensorDataset(torch.ones((4, 4)), torch.ones((4))))
        return train_loader, val_loader

    def get_criterion(self, config: Config) -> _Loss:
        return torch.nn.CrossEntropyLoss()

    def get_optimizer(self, config: Config) -> Optimizer:
        return torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

    def get_model(self, config: Config) -> nn.Module:
        return ToyConvNet().to(self.device)


def test_instance_level_client_with_changes() -> None:
    # Creating client in the right way shouldn't throw an error
    good_client = GoodClientForTest(data_path=Path(""), metrics=[Accuracy()], device=torch.device("cpu"))
    good_client.setup_client({})
    # We should throw an assertion error here because we're trying to use a FedBnClient with the wrong kind of
    # parameter exchanger.
    with pytest.raises(AssertionError):
        bad_client = BadClientForTest(data_path=Path(""), metrics=[Accuracy()], device=torch.device("cpu"))
        bad_client.setup_client({})
