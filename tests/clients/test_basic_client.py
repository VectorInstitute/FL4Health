import datetime
from collections.abc import Sequence
from pathlib import Path
from unittest.mock import MagicMock

import freezegun
import torch
from flwr.common import Scalar
from flwr.common.typing import Config
from freezegun import freeze_time
from torch.utils.data import DataLoader

from fl4health.clients.basic_client import BasicClient
from fl4health.reporting import JsonReporter
from fl4health.reporting.base_reporter import BaseReporter
from fl4health.utils.client import fold_loss_dict_into_metrics
from fl4health.utils.dataset import TensorDataset
from fl4health.utils.logging import LoggingMode
from fl4health.utils.losses import EvaluationLosses
from fl4health.utils.random import set_all_random_seeds, unset_all_random_seeds
from fl4health.utils.typing import TorchInputType, TorchTargetType
from tests.test_utils.assert_metrics_dict import assert_metrics_dict
from tests.test_utils.models_for_test import LinearModel


freezegun.configure(extend_ignore_list=["transformers"])  # type: ignore


def get_dummy_dataset() -> TensorDataset:
    data = torch.randn(100, 10, 8)
    targets = torch.randint(5, (100,))
    return TensorDataset(data=data, targets=targets)


DUMMY_DATASET = get_dummy_dataset()
DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@freeze_time("2012-12-12 12:12:12")
def test_json_reporter_setup_client() -> None:
    reporter = JsonReporter()
    fl_client = MockBasicClient(reporters=[reporter])
    fl_client.setup_client({})

    metric_dict = {
        "host_type": "client",
        "initialized": str(datetime.datetime(2012, 12, 12, 12, 12, 12)),
    }
    errors = assert_metrics_dict(metric_dict, reporter.metrics)
    assert len(errors) == 0, f"Metrics check failed. Errors: {errors}"


@freeze_time("2012-12-12 12:12:12")
def test_json_reporter_shutdown() -> None:
    reporter = JsonReporter()
    fl_client = MockBasicClient(reporters=[reporter])
    fl_client.shutdown()

    metric_dict = {
        "shutdown": str(datetime.datetime(2012, 12, 12, 12, 12, 12)),
    }
    errors = assert_metrics_dict(metric_dict, reporter.metrics)
    assert len(errors) == 0, f"Metrics check failed. Errors: {errors}"


@freeze_time("2012-12-12 12:12:12")
def test_metrics_reporter_fit() -> None:
    test_current_server_round = 2
    test_loss_dict = {"test_loss": 123.123}
    test_metrics: dict[str, Scalar] = {"test_metric": 1234}
    reporter = JsonReporter()

    fl_client = MockBasicClient(loss_dict=test_loss_dict, metrics=test_metrics, reporters=[reporter])
    fl_client.fit([], {"current_server_round": test_current_server_round, "local_epochs": 0})
    metric_dict = {
        "host_type": "client",
        "initialized": str(datetime.datetime(2012, 12, 12, 12, 12, 12)),
        "rounds": {
            test_current_server_round: {
                "round_start": str(datetime.datetime(2012, 12, 12, 12, 12, 12)),
                "fit_round_losses": test_loss_dict,
                "fit_round_metrics": test_metrics,
                "round": test_current_server_round,
            },
        },
    }

    errors = assert_metrics_dict(metric_dict, reporter.metrics)
    assert len(errors) == 0, f"Metrics check failed. Errors: {errors}"


@freeze_time("2012-12-12 12:12:12")
def test_metrics_reporter_evaluate() -> None:
    test_current_server_round = 2
    test_loss = 123.123
    test_metrics: dict[str, Scalar] = {"test_metric": 1234}
    test_metrics_testing: dict[str, Scalar] = {"testing_metric": 1234}
    test_metrics_final = {
        "test_metric": 1234,
        "testing_metric": 1234,
        "val - checkpoint": 123.123,
        "test - checkpoint": 123.123,
        "test - num_examples": 32,
    }
    reporter = JsonReporter()
    fl_client = MockBasicClient(
        loss=test_loss,
        loss_dict={"checkpoint": test_loss},
        metrics=test_metrics,
        test_set_metrics=test_metrics_testing,
        reporters=[reporter],
    )
    fl_client.evaluate(
        [],
        {"current_server_round": test_current_server_round, "local_epochs": 0, "pack_losses_with_val_metrics": True},
    )

    metric_dict = {
        "host_type": "client",
        "initialized": str(datetime.datetime(2012, 12, 12, 12, 12, 12)),
        "rounds": {
            test_current_server_round: {
                "eval_round_start": str(datetime.datetime(2012, 12, 12, 12, 12, 12)),
                "eval_round_loss": test_loss,
                "eval_round_metrics": test_metrics_final,
                "eval_round_end": str(datetime.datetime(2012, 12, 12, 12, 12, 12)),
            },
        },
    }

    errors = assert_metrics_dict(metric_dict, reporter.metrics)
    assert len(errors) == 0, f"Metrics check failed. Errors: {errors}"


def test_evaluate_after_fit_enabled() -> None:
    fl_client = MockBasicClient()
    fl_client.validate = MagicMock()  # type: ignore
    fl_client.validate.return_value = fl_client.mock_loss, fl_client.mock_metrics

    fl_client.fit([], {"current_server_round": 2, "local_epochs": 0, "evaluate_after_fit": True})

    fl_client.validate.assert_called_once()  # type: ignore


def test_evaluate_after_fit_disabled() -> None:
    fl_client = MockBasicClient()
    fl_client.validate = MagicMock()  # type: ignore
    fl_client.validate.return_value = fl_client.mock_loss, fl_client.mock_metrics

    fl_client.fit([], {"current_server_round": 2, "local_epochs": 0, "evaluate_after_fit": False})
    fl_client.validate.assert_not_called()  # type: ignore

    fl_client.fit([], {"current_server_round": 2, "local_epochs": 0})
    fl_client.validate.assert_not_called()  # type: ignore


def test_validate_by_steps() -> None:
    # Set the random seeds
    set_all_random_seeds(2023)

    fl_client = MockBasicClient()
    fl_client.num_validation_steps = 2
    fl_client.model = LinearModel()
    fl_client.device = DEVICE
    fl_client.val_loader = DataLoader(DUMMY_DATASET, batch_size=15, shuffle=False)
    loss, metrics = fl_client._validate_by_steps(fl_client.val_loss_meter, fl_client.val_metric_manager)
    assert loss == (1.0 + 2.0) / 2.0
    assert fl_client.val_iterator is not None

    unset_all_random_seeds()


def test_num_val_samples_correct() -> None:
    fl_client_no_max = MockBasicClient()
    fl_client_no_max.setup_client({})
    assert fl_client_no_max.num_validation_steps is None
    assert fl_client_no_max.num_val_samples == 32

    fl_client_max = MockBasicClient()
    config: Config = {"num_validation_steps": 2}
    fl_client_max.setup_client(config)
    assert fl_client_max.num_validation_steps == 2
    assert fl_client_max.num_val_samples == 8


class MockBasicClient(BasicClient):
    def __init__(
        self,
        loss_dict: dict[str, float] | None = None,
        metrics: dict[str, Scalar] | None = None,
        test_set_metrics: dict[str, Scalar] | None = None,
        loss: float | None = 0,
        reporters: Sequence[BaseReporter] | None = None,
    ):
        super().__init__(Path(""), [], DEVICE, reporters=reporters)

        self.mock_loss_dict = loss_dict
        if self.mock_loss_dict is None:
            self.mock_loss_dict = {}

        self.mock_metrics = metrics
        if self.mock_metrics is None:
            self.mock_metrics = {}

        self.mock_metrics_test = test_set_metrics

        self.mock_loss = loss

        # Mocking attributes
        self.train_loader = MagicMock()
        self.test_loader = MagicMock()
        self.num_train_samples = 0
        self.num_val_samples = 0
        self.num_validation_steps = None

        # Mocking methods
        self.set_parameters = MagicMock()  # type: ignore
        self.get_parameters = MagicMock()  # type: ignore
        self.train_by_epochs = MagicMock()  # type: ignore
        self.train_by_epochs.return_value = self.mock_loss_dict, self.mock_metrics
        self.train_by_steps = MagicMock()  # type: ignore
        self.train_by_steps.return_value = self.mock_loss_dict, self.mock_metrics
        self.get_model = MagicMock()  # type: ignore
        self.get_data_loaders = MagicMock()  # type: ignore
        mock_data_loader = MagicMock()  # type: ignore
        mock_data_loader.batch_size = 4
        mock_data_loader.dataset = [None] * 32
        mock_data_loader.__len__ = lambda _: len(mock_data_loader.dataset) // mock_data_loader.batch_size
        self.get_data_loaders.return_value = mock_data_loader, mock_data_loader
        self.get_test_data_loader = MagicMock()  # type: ignore
        self.get_test_data_loader.return_value = mock_data_loader
        self.get_optimizer = MagicMock()  # type: ignore
        self.get_criterion = MagicMock()  # type: ignore
        self.val_step = MagicMock()  # type: ignore
        self.val_step.side_effect = self.mock_val_step

        self._fully_validate_or_test = MagicMock()  # type: ignore
        self._fully_validate_or_test.side_effect = self.mock_validate_or_test

    def mock_val_step(self, input: TorchInputType, target: TorchTargetType):  # type: ignore
        loss_tensor = torch.randint(1, 4, (1,))
        return EvaluationLosses(loss_tensor), self.mock_metrics

    def mock_validate_or_test(  # type: ignore
        self, loader, loss_meter, metric_manager, logging_mode=LoggingMode.VALIDATION, include_losses_in_metrics=False
    ):
        if include_losses_in_metrics:
            assert self.mock_loss_dict is not None and self.mock_metrics is not None
            fold_loss_dict_into_metrics(self.mock_metrics, self.mock_loss_dict, logging_mode)
        if logging_mode == LoggingMode.VALIDATION:
            return self.mock_loss, self.mock_metrics
        return self.mock_loss, self.mock_metrics_test
