import warnings
from logging import INFO
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
from fl4health.mixins.adaptive_drift_constrained import (
    AdaptiveDriftConstrainedMixin,
    AdaptiveDriftConstrainedProtocol,
    apply_adaptive_drift_to_client,
)
from fl4health.mixins.core_protocols import BasicClientProtocol
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


def test_init() -> None:
    # setup client
    client = _TestAdaptedClient(data_path=Path(""), metrics=[Accuracy()], device=torch.device("cpu"))
    client.model = torch.nn.Linear(5, 5)
    client.optimizers = {"global": torch.optim.SGD(client.model.parameters(), lr=0.0001)}
    client.train_loader = DataLoader(TensorDataset(torch.ones((1000, 28, 28, 1)), torch.ones((1000))))
    client.val_loader = DataLoader(TensorDataset(torch.ones((1000, 28, 28, 1)), torch.ones((1000))))
    client.parameter_exchanger = FullParameterExchangerWithPacking(ParameterPackerAdaptiveConstraint())
    client.initialized = True
    client.setup_client({})

    assert isinstance(client, BasicClientProtocol)
    assert isinstance(client, AdaptiveDriftConstrainedProtocol)


def test_init_raises_value_error_when_basic_client_protocol_not_satisfied() -> None:

    # Create an invalid adapted client such as inheriting the Mixin but nothing else.
    # Since invalid it will raise a warning—see test_subclass_checks_raise_warning
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        class _InvalidTestAdaptedClient(AdaptiveDriftConstrainedMixin):
            pass

    with pytest.raises(RuntimeError, match="This object needs to satisfy `BasicClientProtocolPreSetup`."):

        _InvalidTestAdaptedClient(data_path=Path(""), metrics=[Accuracy()])


def test_when_basic_client_protocol_check_fails_raises_type_error() -> None:
    client = _TestAdaptedClient(data_path=Path(""), metrics=[Accuracy()], device=torch.device("cpu"))

    with pytest.raises(TypeError, match="BasicClientProtocol requirements not met."):
        client.ensure_protocol_compliance()


def test_subclass_checks_raise_no_warning() -> None:

    with warnings.catch_warnings(record=True) as recorded_warnings:

        class _TestInheritanceMixin(AdaptiveDriftConstrainedMixin, _TestBasicClient):
            """subclass should skip validation if is itself a Mixin that inherits AdaptiveDriftConstrainedMixin"""

            pass

        class _DynamicallyCreatedClass(AdaptiveDriftConstrainedMixin, _TestBasicClient):
            """subclass used for dynamic creation of clients with mixins."""

            _dynamically_created = True

    assert len(recorded_warnings) == 0


def test_subclass_checks_raise_warning() -> None:

    msg = (
        "Class _InvalidSubclass inherits from AdaptiveDriftConstrainedMixin but none of its other "
        "base classes is a BasicClient. This may cause runtime errors."
    )
    with pytest.warns(RuntimeWarning, match=msg):

        class _InvalidSubclass(AdaptiveDriftConstrainedMixin):
            """Invalid subclass that warns the user that it expects this class to be mixed with a BasicClient."""

            pass


def test_get_parameters() -> None:
    # setup client
    client = _TestAdaptedClient(data_path=Path(""), metrics=[Accuracy()], device=torch.device("cpu"))
    client.model = torch.nn.Linear(5, 5)
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
    mock_param_exchanger.push_parameters.assert_called_once_with(client.model, config=test_config)
    mock_param_exchanger.pack_parameters.assert_called_once_with(
        push_params_return_model_weights, client.loss_for_adaptation
    )
    assert_array_equal(packed_params, pack_params_return_val)


@patch.object(_TestAdaptedClient, "setup_client")
@patch("fl4health.mixins.adaptive_drift_constrained.FullParameterExchanger")
def test_get_parameters_uninitialized(mock_param_exchanger: MagicMock, mock_setup_client: MagicMock) -> None:
    # setup client
    client = _TestAdaptedClient(data_path=Path(""), metrics=[Accuracy()], device=torch.device("cpu"))
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


@patch("fl4health.mixins.adaptive_drift_constrained.log")
@patch.object(_TestBasicClient, "set_parameters")
def test_set_parameters(mock_super_set_parameters: MagicMock, mock_logger: MagicMock) -> None:
    # setup client
    client = _TestAdaptedClient(data_path=Path(""), metrics=[Accuracy()], device=torch.device("cpu"))
    client.model = torch.nn.Linear(5, 5)
    client.optimizers = {"global": torch.optim.SGD(client.model.parameters(), lr=0.0001)}
    client.train_loader = DataLoader(TensorDataset(torch.ones((1000, 28, 28, 1)), torch.ones((1000))))
    client.val_loader = DataLoader(TensorDataset(torch.ones((1000, 28, 28, 1)), torch.ones((1000))))
    client.parameter_exchanger = FullParameterExchangerWithPacking(ParameterPackerAdaptiveConstraint())
    client.initialized = True
    client.setup_client({})

    # setup mocks
    mock_param_exchanger = MagicMock()
    unpack_params_model_state: NDArray = np.ndarray(shape=(2, 2), dtype=float)
    mock_param_exchanger.unpack_parameters.return_value = (unpack_params_model_state, 0.1)
    client.parameter_exchanger = mock_param_exchanger

    # act
    assert isinstance(client, AdaptiveDriftConstrainedProtocol)
    new_params: NDArray = np.ndarray(shape=(2, 2), dtype=float)
    config: dict[str, Scalar] = {}
    fitting_round = True
    client.set_parameters(new_params, config, fitting_round)

    # assert
    mock_super_set_parameters.assert_called_once_with(unpack_params_model_state, config, fitting_round)
    mock_param_exchanger.unpack_parameters.assert_called_once_with(new_params)
    mock_logger.assert_called_once_with(INFO, "Penalty weight received from the server: 0.1")


def test_dynamically_created_class() -> None:
    adapted_class = apply_adaptive_drift_to_client(_TestBasicClient)

    client = adapted_class(data_path=Path(""), metrics=[Accuracy()], device=torch.device("cpu"))
    client.model = torch.nn.Linear(5, 5)
    client.optimizers = {"global": torch.optim.SGD(client.model.parameters(), lr=0.0001)}
    client.train_loader = DataLoader(TensorDataset(torch.ones((1000, 28, 28, 1)), torch.ones((1000))))  # type: ignore
    client.val_loader = DataLoader(TensorDataset(torch.ones((1000, 28, 28, 1)), torch.ones((1000))))  # type: ignore
    client.parameter_exchanger = FullParameterExchangerWithPacking(ParameterPackerAdaptiveConstraint())
    client.initialized = True
    client.setup_client({})

    assert isinstance(client, BasicClientProtocol)
    assert isinstance(client, AdaptiveDriftConstrainedProtocol)
