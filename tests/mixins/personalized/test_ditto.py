import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn
from flwr.common.typing import Config, NDArray, Scalar
from numpy.testing import assert_array_equal
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
            """subclass should skip validation if is itself a Mixin that inherits DittoPersonalizedMixin"""

            pass

        # attaches _dynamically_created attr
        _ = make_it_personal(_TestBasicClient, "ditto")

    assert len(recorded_warnings) == 0


def test_subclass_checks_raise_warning() -> None:

    # will raise two warnings, one for DittoPersonalizedMixin and another for its super AdaptiveDriftConstrainedMixin
    with pytest.warns((RuntimeWarning, RuntimeWarning)):

        class _InvalidSubclass(DittoPersonalizedMixin):
            """Invalid subclass that warns the user that it expects this class to be mixed with a BasicClient."""

            pass


def test_get_parameters() -> None:
    # setup client
    client = _TestDittoedClient(data_path=Path(""), metrics=[Accuracy()], device=torch.device("cpu"))
    client.model = torch.nn.Linear(5, 5)
    client.global_model = torch.nn.Linear(5, 5)
    client.optimizers = {"global": torch.optim.SGD(client.model.parameters(), lr=0.0001)}

    # setup mocks
    mock_param_exchanger = MagicMock()
    push_params_return_model_weights: NDArray = np.ndarray(shape=(2, 2), dtype=float)
    pack_params_return_val: NDArray = np.ndarray(shape=(2, 2), dtype=float)
    mock_param_exchanger.push_parameters.return_value = push_params_return_model_weights
    mock_param_exchanger.pack_parameters.return_value = pack_params_return_val
    client.parameter_exchanger = mock_param_exchanger
    client.initialized = True

    # act
    test_config: dict[str, Scalar] = {}
    # TODO: fix the mixin/protocol typing that leads to mypy complaint
    packed_params = client.get_parameters(config=test_config)  # type: ignore

    # assert
    mock_param_exchanger.push_parameters.assert_called_once_with(client.global_model, config=test_config)
    mock_param_exchanger.pack_parameters.assert_called_once_with(
        push_params_return_model_weights, client.loss_for_adaptation
    )
    assert_array_equal(packed_params, pack_params_return_val)


@patch.object(_TestDittoedClient, "setup_client")
@patch("fl4health.mixins.personalized.ditto.FullParameterExchanger")
def test_get_parameters_uninitialized(mock_param_exchanger: MagicMock, mock_setup_client: MagicMock) -> None:
    # setup client
    client = _TestDittoedClient(data_path=Path(""), metrics=[Accuracy()], device=torch.device("cpu"))
    client.model = torch.nn.Linear(5, 5)
    client.optimizers = {"global": torch.optim.SGD(client.model.parameters(), lr=0.0001)}
    client.initialized = False

    # setup mocks
    mock_param_exchanger_instance = mock_param_exchanger.return_value
    push_params_return_model_weights: NDArray = np.ndarray(shape=(2, 2), dtype=float)
    mock_param_exchanger_instance.push_parameters.return_value = push_params_return_model_weights

    # act
    test_config: dict[str, Scalar] = {}
    # TODO: fix the mixin/protocol typing that leads to mypy complaint
    packed_params = client.get_parameters(config=test_config)  # type: ignore

    # assert
    mock_setup_client.assert_called_once_with(test_config)
    mock_param_exchanger_instance.push_parameters.assert_called_once_with(client.model, config=test_config)
    assert_array_equal(packed_params, push_params_return_model_weights)
