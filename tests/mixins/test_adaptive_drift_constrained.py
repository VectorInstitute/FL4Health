from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from fl4health.clients.basic_client import BasicClient
from fl4health.metrics import Accuracy
from fl4health.mixins.adaptive_drift_constrained import AdaptiveDriftConstrainedMixin, AdaptiveDriftConstrainedProtocol
from fl4health.mixins.core import BasicClientProtocol
from fl4health.parameter_exchange.packing_exchanger import FullParameterExchangerWithPacking
from fl4health.parameter_exchange.parameter_packer import (
    ParameterPackerAdaptiveConstraint,
)


class _TestBasicClient(BasicClient):
    def get_model(self, config):
        return self.model

    def get_data_loaders(self, config):
        return self.train_loader, self.val_loader

    def get_optimizer(self, config):
        return self.optimizers["global"]

    def get_criterion(self, config):
        return torch.nn.CrossEntropyLoss()


class _TestAdaptedClient(AdaptiveDriftConstrainedMixin, _TestBasicClient):
    pass


def test_init() -> None:
    # setup client
    client = _TestAdaptedClient(data_path=Path(""), metrics=[Accuracy()], device=torch.device("cpu"))
    client.model = torch.nn.Linear(5, 5)
    client.optimizers = {"global": torch.optim.SGD(client.model.parameters(), lr=0.0001)}  # type: ignore
    client.train_loader = DataLoader(TensorDataset(torch.ones((1000, 28, 28, 1)), torch.ones((1000))))  # type: ignore
    client.val_loader = DataLoader(TensorDataset(torch.ones((1000, 28, 28, 1)), torch.ones((1000))))  # type: ignore
    client.parameter_exchanger = FullParameterExchangerWithPacking(ParameterPackerAdaptiveConstraint())

    client.initialized = True

    assert isinstance(client, BasicClientProtocol)
    assert isinstance(client, AdaptiveDriftConstrainedProtocol)
