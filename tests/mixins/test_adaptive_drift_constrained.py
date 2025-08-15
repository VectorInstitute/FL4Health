from contextlib import nullcontext as no_error_raised
from logging import INFO
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from flwr.common.typing import NDArray, Scalar
from numpy.testing import assert_array_equal
from torch.utils.data import DataLoader, TensorDataset

from fl4health.metrics import Accuracy
from fl4health.mixins.adaptive_drift_constrained import (
    AdaptiveDriftConstrainedMixin,
    AdaptiveDriftConstrainedProtocol,
    apply_adaptive_drift_to_client,
)
from fl4health.mixins.core_protocols import FlexibleClientProtocol
from fl4health.parameter_exchange.packing_exchanger import FullParameterExchangerWithPacking
from fl4health.parameter_exchange.parameter_packer import ParameterPackerAdaptiveConstraint
from fl4health.utils.losses import TrainingLosses

from .conftest import _TestFlexibleClient


class _TestAdaptedClient(AdaptiveDriftConstrainedMixin, _TestFlexibleClient):
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

    assert isinstance(client, FlexibleClientProtocol)
    assert isinstance(client, AdaptiveDriftConstrainedProtocol)


def test_subclass_checks_raise_no_error() -> None:
    with no_error_raised():

        class _TestInheritanceMixin(AdaptiveDriftConstrainedMixin, _TestFlexibleClient):
            """Subclass should skip validation if is itself a Mixin that inherits AdaptiveDriftConstrainedMixin."""

            pass

        class _DynamicallyCreatedClass(AdaptiveDriftConstrainedMixin, _TestFlexibleClient):
            """subclass used for dynamic creation of clients with mixins."""

            _dynamically_created = True


def test_subclass_checks_raise_warning_error() -> None:
    msg = (
        "Class _InvalidSubclass inherits from BaseFlexibleMixin but none of its other "
        "base classes implement FlexibleClient."
    )
    with pytest.raises(RuntimeError, match=msg):

        class _InvalidSubclass(AdaptiveDriftConstrainedMixin):
            """Invalid subclass that warns the user that it expects this class to be mixed with a FlexibleClient."""

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
@patch.object(_TestFlexibleClient, "set_parameters")
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
    new_params: NDArray = np.ndarray(shape=(2, 2), dtype=float)
    config: dict[str, Scalar] = {}
    fitting_round = True
    assert isinstance(client, AdaptiveDriftConstrainedProtocol)
    client.set_parameters(new_params, config, fitting_round)

    # assert
    mock_super_set_parameters.assert_called_once_with(unpack_params_model_state, config, fitting_round)
    mock_param_exchanger.unpack_parameters.assert_called_once_with(new_params)
    mock_logger.assert_called_once_with(INFO, "Penalty weight received from the server: 0.1")


def test_dynamically_created_class() -> None:
    adapted_class = apply_adaptive_drift_to_client(_TestFlexibleClient)

    client = adapted_class(data_path=Path(""), metrics=[Accuracy()], device=torch.device("cpu"))
    client.model = torch.nn.Linear(5, 5)
    client.optimizers = {"global": torch.optim.SGD(client.model.parameters(), lr=0.0001)}
    client.train_loader = DataLoader(TensorDataset(torch.ones((1000, 28, 28, 1)), torch.ones((1000))))  # type: ignore
    client.val_loader = DataLoader(TensorDataset(torch.ones((1000, 28, 28, 1)), torch.ones((1000))))  # type: ignore
    client.parameter_exchanger = FullParameterExchangerWithPacking(ParameterPackerAdaptiveConstraint())
    client.initialized = True
    client.setup_client({})

    assert isinstance(client, FlexibleClientProtocol)
    assert isinstance(client, AdaptiveDriftConstrainedProtocol)


@patch.object(_TestAdaptedClient, "_compute_preds_and_losses")
@patch.object(_TestAdaptedClient, "compute_penalty_loss")
@patch.object(_TestAdaptedClient, "_apply_backwards_on_losses_and_take_step")
def test_train_step(
    mock_apply_backwards_on_losses_and_take_step: MagicMock,
    mock_compute_penalty_loss: MagicMock,
    mock_compute_preds_and_losses: MagicMock,
) -> None:
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
    dummy_training_losses = TrainingLosses(
        backward=torch.ones(5),
    )
    dummy_pred_type = {"prediction": torch.ones(5)}
    dummy_penalty_loss = torch.zeros(5)
    mock_compute_preds_and_losses.return_value = (dummy_training_losses, dummy_pred_type)
    mock_compute_penalty_loss.return_value = dummy_penalty_loss
    mock_apply_backwards_on_losses_and_take_step.side_effect = lambda x, y, z: z

    # act
    dummy_input = torch.ones(5)
    dummy_target = torch.zeros(5)
    # TODO: fix the mixin/protocol typing that leads to mypy complaint
    result = client.train_step(dummy_input, dummy_target)  # type: ignore

    # assert
    mock_compute_preds_and_losses.assert_called_once_with(
        client.model, client.optimizers["global"], dummy_input, dummy_target
    )
    mock_compute_penalty_loss.assert_called_once()
    mock_apply_backwards_on_losses_and_take_step.assert_called_once_with(
        client.model, client.optimizers["global"], dummy_training_losses
    )
    assert result[1] == dummy_pred_type


def test_adaptive_client_protocol_attr() -> None:
    """Test interface for AdaptiveDriftConstrainedProtocol."""
    annotations = AdaptiveDriftConstrainedProtocol.__annotations__
    assert "loss_for_adaptation" in annotations
    assert "drift_penalty_tensors" in annotations
    assert "drift_penalty_weight" in annotations
    assert "penalty_loss_function" in annotations
    assert "parameter_exchanger" in annotations

    assert hasattr(AdaptiveDriftConstrainedProtocol, "compute_penalty_loss")
