import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
from unittest.mock import Mock

import torch
from flwr.common import Config, NDArrays, Scalar
from freezegun import freeze_time
from torch import nn as nn
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader

from fl4health.clients.basic_client import BasicClient


@freeze_time("2012-12-12 12:12:12")
def test_metrics_reporter_setup_client() -> None:
    fl_client = MockBasicClient(mock_setup_client=False)
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

    fl_client = MockBasicClient(loss=test_loss, metrics=test_metrics)
    fl_client.evaluate([], {"current_server_round": test_current_server_round, "local_epochs": 0})

    assert fl_client.metrics_reporter.metrics == {
        "rounds": {
            test_current_server_round: {
                "evaluate_start": datetime.datetime(2012, 12, 12, 12, 12, 12),
                "loss": test_loss,
                "evaluate_metrics": test_metrics,
            },
        },
    }


class MockBasicClient(BasicClient):
    def __init__(
        self,
        loss_dict: Optional[Dict[str, float]] = None,
        metrics: Optional[Dict[str, Scalar]] = None,
        loss: Optional[float] = None,
        mock_setup_client: Optional[bool] = True,
    ):
        super().__init__(Path(""), [], torch.device(0)),

        if loss_dict is not None:
            self.loss_dict = loss_dict

        if metrics is not None:
            self.mock_metrics = metrics

        if loss is not None:
            self.loss = loss

        self.mock_setup_client = mock_setup_client

        self.train_loader = []  # type: ignore
        self.num_train_samples = 0
        self.num_val_samples = 0

    def setup_client(self, config: Config) -> None:
        if self.mock_setup_client:
            return
        super().setup_client(config)

    def set_parameters(self, parameters: NDArrays, config: Config) -> None:
        pass

    def get_parameters(self, config: Config) -> NDArrays:
        return []

    def train_by_epochs(
        self, epochs: int, current_round: Optional[int] = None
    ) -> Tuple[Dict[str, float], Dict[str, Scalar]]:
        return self.loss_dict, self.mock_metrics

    def train_by_steps(
        self, steps: int, current_round: Optional[int] = None
    ) -> Tuple[Dict[str, float], Dict[str, Scalar]]:
        return self.loss_dict, self.mock_metrics

    def validate(self) -> Tuple[float, Dict[str, Scalar]]:
        return self.loss, self.mock_metrics

    def get_model(self, config: Config) -> nn.Module:
        return Mock()

    def get_data_loaders(self, config: Config) -> Tuple[DataLoader, ...]:
        mock_data_loader = Mock()
        mock_data_loader.dataset = []
        return mock_data_loader, mock_data_loader

    def set_optimizer(self, config: Config) -> None:
        pass

    def get_criterion(self, config: Config) -> _Loss:
        return Mock()
