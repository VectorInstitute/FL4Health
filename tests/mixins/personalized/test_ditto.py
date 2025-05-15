import warnings
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
from fl4health.mixins.core_protocols import BasicClientProtocol
from fl4health.mixins.personalized import DittoPersonalizedMixin, DittoPersonalizedProtocol, make_it_personal
from fl4health.parameter_exchange.packing_exchanger import FullParameterExchangerWithPacking
from fl4health.parameter_exchange.parameter_packer import (
    ParameterPackerAdaptiveConstraint,
)


class _TestBasicClient(BasicClient):
    def get_model(self, config: Config) -> nn.Module:
        return self.model

    def get_data_loaders(self, config: Config) -> tuple[DataLoader, DataLoader]:
        return self.train_loader, self.val_loader

    def get_optimizer(self, config: Config) -> Optimizer | dict[str, Optimizer]:
        return self.optimizers["global"]

    def get_criterion(self, config: Config) -> _Loss:
        return torch.nn.CrossEntropyLoss()


class _TestDittoedClient(DittoPersonalizedMixin, _TestBasicClient):
    pass


def test_init() -> None:
    # setup client
    client = _TestDittoedClient(data_path=Path(""), metrics=[Accuracy()], device=torch.device("cpu"))
    client.model = torch.nn.Linear(5, 5)
    client.optimizers = {"global": torch.optim.SGD(client.model.parameters(), lr=0.0001)}
    client.train_loader = DataLoader(TensorDataset(torch.ones((1000, 28, 28, 1)), torch.ones((1000))))
    client.val_loader = DataLoader(TensorDataset(torch.ones((1000, 28, 28, 1)), torch.ones((1000))))
    client.parameter_exchanger = FullParameterExchangerWithPacking(ParameterPackerAdaptiveConstraint())
    client.initialized = True
    client.setup_client({})

    assert isinstance(client, BasicClientProtocol)
    assert isinstance(client, DittoPersonalizedProtocol)


# Create an invalid adapted client such as inheriting the Mixin but nothing else.
# Since invalid it will raise a warning—see test_subclass_checks_raise_warning
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_init_raises_value_error_when_basic_client_protocol_not_satisfied() -> None:

    class _InvalidTestDittoClient(DittoPersonalizedMixin):
        pass

    with pytest.raises(RuntimeError, match="This object needs to satisfy `BasicClientProtocolPreSetup`."):

        _InvalidTestDittoClient(data_path=Path(""), metrics=[Accuracy()])


def test_subclass_checks_raise_no_warning() -> None:

    with warnings.catch_warnings(record=True) as recorded_warnings:

        class _TestInheritanceMixin(DittoPersonalizedMixin, _TestBasicClient):
            """subclass should skip validation if is itself a Mixin that inherits AdaptiveDriftConstrainedMixin"""

            pass

        _ = make_it_personal(_TestBasicClient, "ditto")

    assert len(recorded_warnings) == 0
