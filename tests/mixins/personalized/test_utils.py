from contextlib import nullcontext as does_not_raise
from pathlib import Path

import pytest
import torch
from flwr.common.typing import Config
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader, TensorDataset

from fl4health.clients.flexible.base import FlexibleClient
from fl4health.metrics import Accuracy
from fl4health.mixins.personalized.utils import ensure_protocol_compliance
from fl4health.parameter_exchange.packing_exchanger import FullParameterExchangerWithPacking
from fl4health.parameter_exchange.parameter_packer import ParameterPackerAdaptiveConstraint


def test_ensure_protocol_compliance_does_not_raise() -> None:
    # arrange
    class MyClient(FlexibleClient):
        def get_model(self, config: Config) -> nn.Module:
            return self.model

        def get_data_loaders(self, config: Config) -> tuple[DataLoader, DataLoader]:
            return self.train_loader, self.val_loader

        def get_optimizer(self, config: Config) -> Optimizer | dict[str, Optimizer]:
            return self.optimizers["global"]

        def get_criterion(self, config: Config) -> _Loss:
            return torch.nn.CrossEntropyLoss()

        @ensure_protocol_compliance
        def some_method(self, x: int) -> int:
            return x + 1

    # setup client
    client = MyClient(data_path=Path(""), metrics=[Accuracy()], device=torch.device("cpu"))
    client.model = torch.nn.Linear(5, 5)
    client.optimizers = {"global": torch.optim.SGD(client.model.parameters(), lr=0.0001)}
    client.train_loader = DataLoader(TensorDataset(torch.ones((1000, 28, 28, 1)), torch.ones((1000))))
    client.val_loader = DataLoader(TensorDataset(torch.ones((1000, 28, 28, 1)), torch.ones((1000))))
    client.parameter_exchanger = FullParameterExchangerWithPacking(ParameterPackerAdaptiveConstraint())
    client.initialized = True
    client.setup_client({})

    # act/assert
    with does_not_raise():
        client.some_method(2)


def test_ensure_protocol_compliance_does_raise_type_error() -> None:
    # arrange
    class MyClient:
        """My Client DOES not satisfy the protocol of FlexibleClient."""

        @ensure_protocol_compliance
        def some_method(self, x: int) -> int:
            return x + 1

    client = MyClient()

    # act/assert
    with pytest.raises(TypeError, match="Protocol requirements not met."):
        client.some_method(2)
