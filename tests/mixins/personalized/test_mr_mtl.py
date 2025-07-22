import re
from pathlib import Path

import pytest
import torch
from flwr.common.typing import Config
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer

# from torch.testing import assert_close
from torch.utils.data import DataLoader, TensorDataset

from fl4health.clients.flexible.base import FlexibleClient
from fl4health.metrics import Accuracy
from fl4health.mixins.core_protocols import FlexibleClientProtocol
from fl4health.mixins.personalized import (
    MrMtlPersonalizedMixin,
    MrMtlPersonalizedProtocol,
)
from fl4health.parameter_exchange.packing_exchanger import FullParameterExchangerWithPacking
from fl4health.parameter_exchange.parameter_packer import ParameterPackerAdaptiveConstraint


class _TestFlexibleClient(FlexibleClient):
    def get_model(self, config: Config) -> nn.Module:
        return self.model

    def get_data_loaders(self, config: Config) -> tuple[DataLoader, DataLoader]:
        return self.train_loader, self.val_loader

    def get_optimizer(self, config: Config) -> Optimizer | dict[str, Optimizer]:
        return self.optimizers["local"]

    def get_criterion(self, config: Config) -> _Loss:
        return torch.nn.CrossEntropyLoss()


class _TestDittoedClient(MrMtlPersonalizedMixin, _TestFlexibleClient):
    pass


class _DummyParent:
    def __init__(self) -> None:
        pass


class _TestInvalidMrMtlClient(MrMtlPersonalizedMixin, _DummyParent):
    pass


def test_raise_runtime_error_not_flexible_client() -> None:
    """Test that an invalid parent raises RuntimeError."""
    with pytest.raises(
        RuntimeError, match=re.escape("This object needs to satisfy `FlexibleClientProtocolPreSetup`.")
    ):
        _TestInvalidMrMtlClient()


def test_init() -> None:
    # setup client
    client = _TestDittoedClient(data_path=Path(""), metrics=[Accuracy()], device=torch.device("cpu"))
    client.model = torch.nn.Linear(5, 5)
    client.optimizers = {"local": torch.optim.SGD(client.model.parameters(), lr=0.0001)}
    client.train_loader = DataLoader(TensorDataset(torch.ones((1000, 28, 28, 1)), torch.ones((1000))))
    client.val_loader = DataLoader(TensorDataset(torch.ones((1000, 28, 28, 1)), torch.ones((1000))))
    client.parameter_exchanger = FullParameterExchangerWithPacking(ParameterPackerAdaptiveConstraint())
    client.initialized = True
    client.setup_client({})

    assert isinstance(client, FlexibleClientProtocol)
    assert isinstance(client, MrMtlPersonalizedProtocol)
