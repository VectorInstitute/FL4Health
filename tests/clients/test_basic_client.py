import datetime
from pathlib import Path
from typing import Dict, Optional, Union
from unittest.mock import MagicMock

import freezegun
import torch
from flwr.common import Scalar
from freezegun import freeze_time

from fl4health.clients.basic_client import BasicClient

freezegun.configure(extend_ignore_list=["transformers"])  # type: ignore


@freeze_time("2012-12-12 12:12:12")
def test_metrics_reporter_setup_client() -> None:
    fl_client = MockBasicClient()
    fl_client.setup_client({})

    assert fl_client.metrics_reporter.metrics == {
        "type": "client",
        "initialized": datetime.datetime(2012, 12, 12, 12, 12, 12),
    }


@freeze_time("2012-12-12 12:12:12")
def test_metrics_reporter_shutdown() -> None:
    fl_client = MockBasicClient()
    fl_client.shutdown()

    assert fl_client.metrics_reporter.metrics == {
        "shutdown": datetime.datetime(2012, 12, 12, 12, 12, 12),
    }


@freeze_time("2012-12-12 12:12:12")
def test_metrics_reporter_fit() -> None:
    test_current_server_round = 2
    test_loss_dict = {"test_loss": 123.123}
    test_metrics: Dict[str, Union[bool, bytes, float, int, str]] = {"test_metric": 1234}

    fl_client = MockBasicClient(loss_dict=test_loss_dict, metrics=test_metrics)
    fl_client.fit([], {"current_server_round": test_current_server_round, "local_epochs": 0})

    assert fl_client.metrics_reporter.metrics == {
        "type": "client",
        "initialized": datetime.datetime(2012, 12, 12, 12, 12, 12),
        "rounds": {
            test_current_server_round: {
                "fit_start": datetime.datetime(2012, 12, 12, 12, 12, 12),
                "loss_dict": test_loss_dict,
                "fit_metrics": test_metrics,
            },
        },
    }


@freeze_time("2012-12-12 12:12:12")
def test_metrics_reporter_evaluate() -> None:
    test_current_server_round = 2
    test_loss = 123.123
    test_metrics: Dict[str, Union[bool, bytes, float, int, str]] = {"test_metric": 1234}
    final_test_metrics = {"test_metric": 1234, "test - loss": 123.123}

    fl_client = MockBasicClient(loss=test_loss, metrics=test_metrics)
    fl_client.evaluate([], {"current_server_round": test_current_server_round, "local_epochs": 0})

    assert fl_client.metrics_reporter.metrics == {
        "type": "client",
        "initialized": datetime.datetime(2012, 12, 12, 12, 12, 12),
        "rounds": {
            test_current_server_round: {
                "evaluate_start": datetime.datetime(2012, 12, 12, 12, 12, 12),
                "loss": test_loss,
                "evaluate_metrics": final_test_metrics,
            },
        },
    }


def test_evaluate_after_fit_enabled() -> None:
    fl_client = MockBasicClient()

    fl_client.fit([], {"current_server_round": 2, "local_epochs": 0, "evaluate_after_fit": True})

    fl_client.validate.assert_called_once()  # type: ignore


def test_evaluate_after_fit_disabled() -> None:
    fl_client = MockBasicClient()

    fl_client.fit([], {"current_server_round": 2, "local_epochs": 0, "evaluate_after_fit": False})
    fl_client.validate.assert_not_called()  # type: ignore

    fl_client.fit([], {"current_server_round": 2, "local_epochs": 0})
    fl_client.validate.assert_not_called()  # type: ignore


class MockBasicClient(BasicClient):
    def __init__(
        self,
        loss_dict: Optional[Dict[str, float]] = None,
        metrics: Optional[Dict[str, Scalar]] = None,
        loss: Optional[float] = 0,
    ):
        super().__init__(Path(""), [], torch.device(0))

        self.mock_loss_dict = loss_dict
        if self.mock_loss_dict is None:
            self.mock_loss_dict = {}

        self.mock_metrics = metrics
        if self.mock_metrics is None:
            self.mock_metrics = {}

        self.mock_loss = loss

        # Mocking attributes
        self.train_loader = MagicMock()
        self.test_loader = MagicMock()
        self.num_train_samples = 0
        self.num_val_samples = 0

        # Mocking methods
        self.set_parameters = MagicMock()  # type: ignore
        self.get_parameters = MagicMock()  # type: ignore
        self.train_by_epochs = MagicMock()  # type: ignore
        self.train_by_epochs.return_value = self.mock_loss_dict, self.mock_metrics
        self.train_by_steps = MagicMock()  # type: ignore
        self.train_by_steps.return_value = self.mock_loss_dict, self.mock_metrics
        self.validate = MagicMock()  # type: ignore
        self.validate.return_value = self.mock_loss, self.mock_metrics
        self.testing = MagicMock()  # type: ignore
        self.testing.return_value = self.mock_loss, self.mock_metrics
        self.get_model = MagicMock()  # type: ignore
        self.get_data_loaders = MagicMock()  # type: ignore
        mock_data_loader = MagicMock()  # type: ignore
        mock_data_loader.dataset = []
        self.get_data_loaders.return_value = mock_data_loader, mock_data_loader
        self.get_test_data_loader = MagicMock()  # type: ignore
        self.get_test_data_loader.return_value = mock_data_loader
        self.get_optimizer = MagicMock()  # type: ignore
        self.get_criterion = MagicMock()  # type: ignore
