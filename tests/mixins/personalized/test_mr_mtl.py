import re
from pathlib import Path

import pytest
import torch

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

from .conftest import _DummyParent, _TestFlexibleClient


class _TestMrMtlPersonalizedClient(MrMtlPersonalizedMixin, _TestFlexibleClient):
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
