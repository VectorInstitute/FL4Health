import warnings
from logging import INFO
from pathlib import Path
from unittest.mock import MagicMock, _Call, patch

import numpy as np
import pytest
import torch
import torch.nn as nn
from flwr.common.typing import Config, NDArray, Scalar
from numpy.testing import assert_array_equal
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.testing import assert_close
from torch.utils.data import DataLoader, TensorDataset

from fl4health.clients.basic_client import BasicClient
from fl4health.metrics import Accuracy
from fl4health.mixins.core_protocols import BasicClientProtocol
from fl4health.mixins.personalized import (
    DittoPersonalizedMixin,
    DittoPersonalizedProtocol,
    PersonalizedMode,
    make_it_personal,
)
from fl4health.parameter_exchange.packing_exchanger import FullParameterExchangerWithPacking
from fl4health.parameter_exchange.parameter_packer import (
    ParameterPackerAdaptiveConstraint,
)
from fl4health.utils.losses import TrainingLosses
from fl4health.utils.typing import TorchFeatureType, TorchInputType, TorchPredType


class _TestBasicClient(BasicClient):
    def get_model(self, config: Config) -> nn.Module:
        return self.model

    def get_data_loaders(self, config: Config) -> tuple[DataLoader, DataLoader]:
        return self.train_loader, self.val_loader

    def get_optimizer(self, config: Config) -> Optimizer | dict[str, Optimizer]:
        return self.optimizers["local"]

    def get_criterion(self, config: Config) -> _Loss:
        return torch.nn.CrossEntropyLoss()


class _TestBasicClientV2(BasicClient):
    def get_model(self, config: Config) -> nn.Module:
        return self.model

    def get_data_loaders(self, config: Config) -> tuple[DataLoader, DataLoader]:
        return self.train_loader, self.val_loader

    def get_optimizer(self, config: Config) -> Optimizer | dict[str, Optimizer]:
        return self.optimizers["local"]

    def get_criterion(self, config: Config) -> _Loss:
        return torch.nn.CrossEntropyLoss()

    def _predict(self, model: torch.nn.Module, input: TorchInputType) -> tuple[TorchPredType, TorchFeatureType]:
        return {}, {}


class _TestDittoedClient(DittoPersonalizedMixin, _TestBasicClient):
    pass


class _TestDittoedClientV2(DittoPersonalizedMixin, _TestBasicClientV2):
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
        _ = make_it_personal(_TestBasicClient, PersonalizedMode.DITTO)

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


@patch.object(_TestDittoedClient, "setup_client")
@patch("fl4health.mixins.personalized.ditto.FullParameterExchanger")
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


def test_predict() -> None:
    # setup client
    client = _TestDittoedClient(data_path=Path(""), metrics=[Accuracy()], device=torch.device("cpu"))

    mock_model = MagicMock()
    mock_global_model = MagicMock()

    mock_model.return_value = torch.ones(5)
    mock_global_model.return_value = torch.zeros(5)

    client.model = mock_model
    client.global_model = mock_global_model

    client.optimizers = {
        "global": MagicMock(),
        "local": MagicMock(),
    }

    client.train_loader = DataLoader(TensorDataset(torch.ones((1000, 28, 28, 1)), torch.ones((1000))))
    client.val_loader = DataLoader(TensorDataset(torch.ones((1000, 28, 28, 1)), torch.ones((1000))))
    client.parameter_exchanger = FullParameterExchangerWithPacking(ParameterPackerAdaptiveConstraint())
    client.initialized = True

    # act
    # TODO: fix the mixin/protocol typing that leads to mypy complaint
    res, _ = client.predict(input=torch.zeros(5))  # type: ignore
    print(f"res: {res}")
    print(torch.zeros(5))

    # assert
    assert_close(res["global"], torch.zeros(5))
    assert_close(res["local"], torch.ones(5))


@patch.object(_TestDittoedClientV2, "_predict")
def test_predict_delagation(private_predict: MagicMock) -> None:
    # setup client
    client = _TestDittoedClientV2(data_path=Path(""), metrics=[Accuracy()], device=torch.device("cpu"))
    client.model = torch.nn.Linear(5, 5)
    client.global_model = torch.nn.Linear(5, 5)

    private_predict.side_effect = [(torch.zeros(5), {}), (torch.ones(5), {})]

    client.optimizers = {
        "global": MagicMock(),
        "local": MagicMock(),
    }

    client.train_loader = DataLoader(TensorDataset(torch.ones((1000, 28, 28, 1)), torch.ones((1000))))
    client.val_loader = DataLoader(TensorDataset(torch.ones((1000, 28, 28, 1)), torch.ones((1000))))
    client.parameter_exchanger = FullParameterExchangerWithPacking(ParameterPackerAdaptiveConstraint())
    client.initialized = True

    # act
    # TODO: fix the mixin/protocol typing that leads to mypy complaint
    res, _ = client.predict(input=torch.zeros(5))  # type: ignore
    print(f"res: {res}")
    print(torch.zeros(5))

    # assert
    assert_close(res["global"], torch.zeros(5))
    assert_close(res["local"], torch.ones(5))


def test_extract_pred() -> None:
    # setup client
    client = _TestDittoedClient(data_path=Path(""), metrics=[Accuracy()], device=torch.device("cpu"))

    res = client._extract_pred(
        kind="global", preds={"global-xyz": torch.ones(5), "global-abc": torch.zeros(5), "local": torch.zeros(5)}
    )

    assert_close(res["xyz"], torch.ones(5))
    assert_close(res["abc"], torch.zeros(5))


def test_extract_pred_raises_error() -> None:
    # setup client
    client = _TestDittoedClient(data_path=Path(""), metrics=[Accuracy()], device=torch.device("cpu"))

    with pytest.raises(ValueError):
        client._extract_pred(
            kind="oops", preds={"global-xyz": torch.ones(5), "global-abc": torch.zeros(5), "local": torch.zeros(5)}
        )


@patch.object(_TestDittoedClient, "set_initial_global_tensors")
@patch.object(_TestBasicClient, "update_before_train")
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


@patch("torch.optim.Optimizer")
@patch.object(_TestDittoedClient, "predict")
@patch.object(_TestDittoedClient, "compute_training_loss")
def test_train_step(
    mock_compute_training_loss: MagicMock, mock_predict: MagicMock, mock_optimizer_class: MagicMock
) -> None:
    # setup client
    client = _TestDittoedClient(data_path=Path(""), metrics=[Accuracy()], device=torch.device("cpu"))
    client.optimizers = {
        "global": MagicMock(),
        "local": MagicMock(),
    }
    client.train_loader = DataLoader(TensorDataset(torch.ones((1000, 28, 28, 1)), torch.ones((1000))))
    client.val_loader = DataLoader(TensorDataset(torch.ones((1000, 28, 28, 1)), torch.ones((1000))))
    client.parameter_exchanger = FullParameterExchangerWithPacking(ParameterPackerAdaptiveConstraint())
    client.initialized = True

    # arrange mocks
    pred_return: tuple[TorchPredType, TorchFeatureType] = {"local": torch.ones(5)}, {}
    preds, features = pred_return
    mock_predict.return_value = preds, features
    mock_backward_loss = MagicMock()
    mock_additional_global_loss = MagicMock()
    losses = TrainingLosses(
        backward={"backward": mock_backward_loss}, additional_losses={"global_loss": mock_additional_global_loss}
    )
    mock_compute_training_loss.return_value = losses

    # act
    input, target = torch.tensor([1, 1, 1, 1, 1]), torch.zeros(3)
    # TODO: fix the mixin/protocol typing that leads to mypy complaint
    retval = client.train_step(input, target)  # type: ignore

    mock_predict.assert_called_once()
    client.optimizers["global"].zero_grad.assert_called_once()
    client.optimizers["local"].zero_grad.assert_called_once()
    mock_compute_training_loss.assert_called_once_with(preds, features, target)
    mock_backward_loss.backward.assert_called_once()
    mock_additional_global_loss.backward.assert_called_once()
    assert retval == (losses, preds)
