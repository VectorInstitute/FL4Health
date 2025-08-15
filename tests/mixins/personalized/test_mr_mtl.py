from contextlib import nullcontext as no_error_raised
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from flwr.common.typing import Config

# from torch.testing import assert_close
from torch.utils.data import DataLoader, TensorDataset

from fl4health.metrics import Accuracy
from fl4health.mixins.core_protocols import FlexibleClientProtocol
from fl4health.mixins.personalized import (
    MrMtlPersonalizedMixin,
    MrMtlPersonalizedProtocol,
)
from fl4health.parameter_exchange.packing_exchanger import FullParameterExchangerWithPacking
from fl4health.parameter_exchange.parameter_packer import ParameterPackerAdaptiveConstraint

from ..conftest import _DummyParent, _TestFlexibleClient


class _TestMrMtlPersonalizedClient(MrMtlPersonalizedMixin, _TestFlexibleClient):
    pass


def test_subclass_checks_raise_no_error() -> None:
    with no_error_raised():

        class _TestMrMtlPersonalizedClient(MrMtlPersonalizedMixin, _TestFlexibleClient):
            pass


def test_subclass_checks_raise_error() -> None:
    msg = (
        "Class _TestInvalidMrMtlClient inherits from BaseFlexibleMixin but none of its other "
        "base classes implement FlexibleClient."
    )
    with pytest.raises(RuntimeError, match=msg):

        class _TestInvalidMrMtlClient(MrMtlPersonalizedMixin, _DummyParent):
            pass


def test_init() -> None:
    # setup client
    client = _TestMrMtlPersonalizedClient(data_path=Path(""), metrics=[Accuracy()], device=torch.device("cpu"))
    client.model = torch.nn.Linear(5, 5)
    client.optimizers = {"local": torch.optim.SGD(client.model.parameters(), lr=0.0001)}
    client.train_loader = DataLoader(TensorDataset(torch.ones((1000, 28, 28, 1)), torch.ones((1000))))
    client.val_loader = DataLoader(TensorDataset(torch.ones((1000, 28, 28, 1)), torch.ones((1000))))
    client.parameter_exchanger = FullParameterExchangerWithPacking(ParameterPackerAdaptiveConstraint())
    client.initialized = True
    client.setup_client({})

    assert isinstance(client, FlexibleClientProtocol)
    assert isinstance(client, MrMtlPersonalizedProtocol)


@patch.object(_TestMrMtlPersonalizedClient, "get_global_model")
def test_get_optimizer(get_global_model: MagicMock) -> None:
    # setup client
    client = _TestMrMtlPersonalizedClient(data_path=Path(""), metrics=[Accuracy()], device=torch.device("cpu"))
    assert client.initial_global_model is None
    mock_global_model = "mock-global-model"
    get_global_model.return_value = mock_global_model
    client.model = torch.nn.Linear(5, 5)
    client.optimizers = {"local": torch.optim.SGD(client.model.parameters(), lr=0.0001)}
    client.train_loader = DataLoader(TensorDataset(torch.ones((1000, 28, 28, 1)), torch.ones((1000))))
    client.val_loader = DataLoader(TensorDataset(torch.ones((1000, 28, 28, 1)), torch.ones((1000))))
    client.parameter_exchanger = FullParameterExchangerWithPacking(ParameterPackerAdaptiveConstraint())

    # act
    mock_config: Config = {}
    _ = client.get_optimizer(mock_config)

    # assert
    get_global_model.assert_called_once_with(mock_config)
    assert client.initial_global_model == mock_global_model
