from contextlib import nullcontext as no_error_raised
from logging import INFO
from pathlib import Path
from unittest.mock import MagicMock, _Call, patch

import numpy as np
import pytest
import torch
from flwr.common.typing import NDArray, Scalar
from numpy.testing import assert_array_equal

# from torch.testing import assert_close
from torch.utils.data import DataLoader, TensorDataset

from fl4health.metrics import Accuracy
from fl4health.mixins.core_protocols import FlexibleClientProtocol
from fl4health.mixins.personalized import (
    DittoPersonalizedMixin,
    DittoPersonalizedProtocol,
    PersonalizedMode,
    make_it_personal,
)
from fl4health.parameter_exchange.packing_exchanger import FullParameterExchangerWithPacking
from fl4health.parameter_exchange.parameter_packer import ParameterPackerAdaptiveConstraint
from fl4health.utils.losses import EvaluationLosses, TrainingLosses

from ..conftest import _DummyParent, _TestFlexibleClient


class _TestDittoedClient(DittoPersonalizedMixin, _TestFlexibleClient):
    pass


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
    assert isinstance(client, DittoPersonalizedProtocol)


def test_subclass_checks_raise_no_error() -> None:
    with no_error_raised():

        class _TestInheritanceMixin(DittoPersonalizedMixin, _TestFlexibleClient):
            """Subclass should skip validation if is itself a Mixin that inherits DittoPersonalizedMixin."""

            pass

        # attaches _dynamically_created attr
        _ = make_it_personal(_TestFlexibleClient, PersonalizedMode.DITTO)


def test_subclass_checks_raise_error() -> None:
    msg = (
        "Class _TestInvalidDittoedClient inherits from BaseFlexibleMixin but none of its other "
        "base classes implement FlexibleClient."
    )
    with pytest.raises(RuntimeError, match=msg):

        class _TestInvalidDittoedClient(DittoPersonalizedMixin, _DummyParent):
            pass


def test_get_parameters() -> None:
    # setup client
    client = _TestDittoedClient(data_path=Path(""), metrics=[Accuracy()], device=torch.device("cpu"))
    client.model = torch.nn.Linear(5, 5)
    client.global_model = torch.nn.Linear(5, 5)
    client.optimizers = {"local": torch.optim.SGD(client.model.parameters(), lr=0.0001)}

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


@patch.object(_TestDittoedClient, "compute_penalty_loss")
@patch.object(_TestDittoedClient, "_apply_backwards_on_losses_and_take_step")
@patch.object(_TestDittoedClient, "_compute_preds_and_losses")
def test_train_step(
    mock_private_compute_preds_and_losses: MagicMock,
    mock_private_apply_backwards_on_losses_and_take_step: MagicMock,
    mock_compute_penalty_loss: MagicMock,
) -> None:
    # setup client
    client = _TestDittoedClient(data_path=Path(""), metrics=[Accuracy()], device=torch.device("cpu"))
    client.model = torch.nn.Linear(5, 5)
    client.global_model = torch.nn.Linear(5, 5)
    client.optimizers = {
        "global": torch.optim.SGD(client.model.parameters(), lr=0.0001),
        "local": torch.optim.SGD(client.model.parameters(), lr=0.0001),
    }
    # setup mocks
    mock_param_exchanger = MagicMock()
    push_params_return_model_weights: NDArray = np.ndarray(shape=(2, 2), dtype=float)
    pack_params_return_val: NDArray = np.ndarray(shape=(2, 2), dtype=float)
    mock_param_exchanger.push_parameters.return_value = push_params_return_model_weights
    mock_param_exchanger.pack_parameters.return_value = pack_params_return_val
    client.parameter_exchanger = mock_param_exchanger
    client.initialized = True

    mock_backward_loss = MagicMock()
    mock_additional_global_loss = MagicMock()
    dummy_training_losses = TrainingLosses(
        backward={"backward": mock_backward_loss}, additional_losses={"global_loss": mock_additional_global_loss}
    )
    dummy_training_losses_for_local = TrainingLosses(
        backward={"backward": mock_backward_loss}, additional_losses={"global_loss": mock_additional_global_loss}
    )
    mock_private_compute_preds_and_losses.side_effect = [
        (
            dummy_training_losses,
            {"prediction": torch.Tensor([1, 2, 3, 4, 5])},
        ),
        (
            dummy_training_losses_for_local,
            {"prediction": torch.Tensor([1, 2, 3, 4, 5])},
        ),
    ]
    mock_private_apply_backwards_on_losses_and_take_step.side_effect = [
        dummy_training_losses,
        dummy_training_losses_for_local,
    ]

    # act
    input, target = torch.tensor([1, 1, 1, 1, 1]), torch.zeros(3)
    # TODO: fix the mixin/protocol typing that leads to mypy complaint
    _ = client.train_step(input, target)  # type: ignore

    mock_private_compute_preds_and_losses.assert_has_calls(
        [
            _Call(((client.global_model, client.optimizers["global"], input, target), {})),
            _Call(((client.model, client.optimizers["local"], input, target), {})),
        ]
    )
    mock_private_apply_backwards_on_losses_and_take_step.assert_has_calls(
        [
            _Call(((client.global_model, client.optimizers["global"], dummy_training_losses), {})),
            _Call(((client.model, client.optimizers["local"], dummy_training_losses_for_local), {})),
        ]
    )
    mock_compute_penalty_loss.assert_called_once()


@patch.object(_TestDittoedClient, "setup_client")
@patch("fl4health.mixins.adaptive_drift_constrained.FullParameterExchanger")
def test_get_parameters_uninitialized(mock_param_exchanger: MagicMock, mock_setup_client: MagicMock) -> None:
    # setup client
    client = _TestDittoedClient(data_path=Path(""), metrics=[Accuracy()], device=torch.device("cpu"))
    client.model = torch.nn.Linear(5, 5)
    client.optimizers = {"local": torch.optim.SGD(client.model.parameters(), lr=0.0001)}
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


@patch("fl4health.mixins.personalized.ditto.log")
def test_set_parameters(mock_logger: MagicMock) -> None:
    # setup client
    client = _TestDittoedClient(data_path=Path(""), metrics=[Accuracy()], device=torch.device("cpu"))
    client.model = torch.nn.Linear(5, 5)
    client.optimizers = {
        "global": torch.optim.SGD(client.model.parameters(), lr=0.0001),
        "local": torch.optim.SGD(client.model.parameters(), lr=0.0001),
    }
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
    config: dict[str, Scalar] = {"current_server_round": 1}
    fitting_round = True
    assert isinstance(client, DittoPersonalizedProtocol)
    client.set_parameters(new_params, config, fitting_round)

    # assert
    mock_param_exchanger.unpack_parameters.assert_called_once_with(new_params)
    mock_logger.assert_has_calls(
        [
            _Call(((INFO, "global model set: Linear"), {})),
            _Call(((INFO, "Lambda weight received from the server: 0.1"), {})),
            _Call(((INFO, "Initializing the global and local models weights for the first time"), {})),
        ]
    )


@patch.object(_TestDittoedClient, "_copy_optimizer_with_new_params")
def test_get_optimizer(mock_copy_optimizer: MagicMock) -> None:
    # setup client
    client = _TestDittoedClient(data_path=Path(""), metrics=[Accuracy()], device=torch.device("cpu"))
    client.model = torch.nn.Linear(5, 5)
    client.optimizers = {"local": torch.optim.SGD(client.model.parameters(), lr=0.0001)}
    client.train_loader = DataLoader(TensorDataset(torch.ones((1000, 28, 28, 1)), torch.ones((1000))))
    client.val_loader = DataLoader(TensorDataset(torch.ones((1000, 28, 28, 1)), torch.ones((1000))))
    client.parameter_exchanger = FullParameterExchangerWithPacking(ParameterPackerAdaptiveConstraint())

    copied_optimizer = torch.optim.SGD(client.model.parameters(), lr=0.0001)
    mock_copy_optimizer.return_value = copied_optimizer

    # act
    optimizers = client.get_optimizer({})

    # assert
    assert client.optimizers["local"] == optimizers["local"]
    assert optimizers["global"] == copied_optimizer
    mock_copy_optimizer.assert_called_once_with(client.optimizers["local"])


@patch.object(_TestDittoedClient, "set_initial_global_tensors")
@patch.object(_TestFlexibleClient, "update_before_train")
def test_update_before_train(
    mock_super_update_before_train: MagicMock, mock_set_initial_global_tensors: MagicMock
) -> None:
    client = _TestDittoedClient(data_path=Path(""), metrics=[Accuracy()], device=torch.device("cpu"))
    mock_global_model = MagicMock()
    client.global_model = mock_global_model

    # act
    client.update_before_train(1)

    mock_global_model.train.assert_called_once()
    mock_super_update_before_train.assert_called_once_with(1)
    mock_set_initial_global_tensors.assert_called_once()


def test_safe_model_raises_error() -> None:
    client = _TestDittoedClient(data_path=Path(""), metrics=[Accuracy()], device=torch.device("cpu"))

    with pytest.raises(ValueError):
        # TODO: fix the mixin/protocol typing that leads to mypy complaint
        client.safe_global_model()  # type: ignore


@patch.object(_TestDittoedClient, "_val_step_with_model")
def test_val_step(
    mock_private_val_step_with_model: MagicMock,
) -> None:
    # setup client
    client = _TestDittoedClient(data_path=Path(""), metrics=[Accuracy()], device=torch.device("cpu"))
    client.model = torch.nn.Linear(5, 5)
    client.global_model = torch.nn.Linear(5, 5)
    client.optimizers = {
        "global": torch.optim.SGD(client.model.parameters(), lr=0.0001),
        "local": torch.optim.SGD(client.model.parameters(), lr=0.0001),
    }
    # setup mocks
    mock_param_exchanger = MagicMock()
    push_params_return_model_weights: NDArray = np.ndarray(shape=(2, 2), dtype=float)
    pack_params_return_val: NDArray = np.ndarray(shape=(2, 2), dtype=float)
    mock_param_exchanger.push_parameters.return_value = push_params_return_model_weights
    mock_param_exchanger.pack_parameters.return_value = pack_params_return_val
    client.parameter_exchanger = mock_param_exchanger
    client.initialized = True

    dummy_training_losses = EvaluationLosses(
        checkpoint=torch.ones(5),
    )
    dummy_training_losses_for_local = EvaluationLosses(
        checkpoint=torch.ones(5),
    )
    mock_private_val_step_with_model.side_effect = [
        (
            dummy_training_losses,
            {"prediction": torch.Tensor([1, 2, 3, 4, 5])},
        ),
        (
            dummy_training_losses_for_local,
            {"prediction": torch.Tensor([1, 2, 3, 4, 5])},
        ),
    ]

    # act
    input, target = torch.tensor([1, 1, 1, 1, 1]), torch.zeros(3)
    # TODO: fix the mixin/protocol typing that leads to mypy complaint
    _ = client.val_step(input, target)  # type: ignore

    mock_private_val_step_with_model.assert_has_calls(
        [
            _Call(((client.global_model, input, target), {})),
            _Call(((client.model, input, target), {})),
        ]
    )
