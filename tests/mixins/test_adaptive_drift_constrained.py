from pathlib import Path

import pytest
import torch
import torch.nn as nn
from flwr.common.typing import Config
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
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
    def get_model(self, config: Config) -> nn.Module:
        return self.model

    def get_data_loaders(self, config: Config) -> tuple[DataLoader, ...]:
        return self.train_loader, self.val_loader

    def get_optimizer(self, config: Config) -> Optimizer | dict[str, Optimizer]:
        return self.optimizers["global"]

    def get_criterion(self, config: Config) -> _Loss:
        return torch.nn.CrossEntropyLoss()


class _TestAdaptedClient(AdaptiveDriftConstrainedMixin, _TestBasicClient):
    pass


class _InvalidTestAdaptedClient(AdaptiveDriftConstrainedMixin):
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
    client.setup_client({})

    assert isinstance(client, BasicClientProtocol)
    assert isinstance(client, AdaptiveDriftConstrainedProtocol)


def test_init_raises_value_error_when_basic_client_protocol_not_satisfied() -> None:
    with pytest.raises(RuntimeError, match="This object needs to satisfy `BasicClientProtocolPreSetup`."):

        _InvalidTestAdaptedClient(data_path=Path(""), metrics=[Accuracy()])


def test_when_basic_client_protocol_check_fails_raises_type_error() -> None:
    client = _TestAdaptedClient(data_path=Path(""), metrics=[Accuracy()], device=torch.device("cpu"))

    with pytest.raises(TypeError, match="BasicClientProtocol requirements not met."):
        client.ensure_protocol_compliance()
