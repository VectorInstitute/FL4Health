from collections.abc import Sequence
from pathlib import Path
from typing import Dict, Optional
from unittest.mock import MagicMock

import torch
from flwr.common import Scalar

from fl4health.clients.basic_client import BasicClient
from fl4health.reporting.base_reporter import BaseReporter
from fl4health.utils.client import fold_loss_dict_into_metrics
from fl4health.utils.logging import LoggingMode
from tests.test_utils.assert_metrics_dict import assert_metrics_dict
from fl4health.reporting.reports_manager import ReportsManager
from fl4health.utils.losses import TrainingLosses, LossMeter
from fl4health.utils.metrics import MetricManager
from fl4health.reporting import JsonReporter
from fl4health.utils.snapshotter import SerizableObjectSnapshotter
import copy


def test_loss_meter_snapshotter() -> None:
    metrics: Dict[str, Scalar] = {"test_metric": 1234}
    reporter = JsonReporter()
    fl_client = MockBasicClient(metrics=metrics,reporters=[reporter])
    ckpt = {}

    fl_client.train_loss_meter.update( TrainingLosses(backward=torch.Tensor([35]), additional_losses=None))
    snapshotter = SerizableObjectSnapshotter(fl_client)
    ckpt['train_loss_meter'] = snapshotter.save("train_loss_meter", LossMeter)
    old_loss_meter = copy.deepcopy(fl_client.train_loss_meter)
    fl_client.train_loss_meter.update(TrainingLosses(backward=torch.Tensor([10]), additional_losses=None))
    assert len(old_loss_meter.losses_list) != len(fl_client.train_loss_meter.losses_list)

    snapshotter.load(ckpt, "train_loss_meter", LossMeter)

    assert len(old_loss_meter.losses_list) == len(fl_client.train_loss_meter.losses_list)
    for i in range(len(fl_client.train_loss_meter.losses_list)):
        assert old_loss_meter.losses_list[i].backward == fl_client.train_loss_meter.losses_list[i].backward
        assert old_loss_meter.losses_list[i].additional_losses == fl_client.train_loss_meter.losses_list[i].additional_losses

def test_reports_manager_snapshotter() -> None:
    metrics: Dict[str, Scalar] = {"test_metric": 1234}
    reporter = JsonReporter()
    fl_client = MockBasicClient(metrics=metrics,reporters=[reporter])
    ckpt = {}

    fl_client.reports_manager.report({"start": "2012-12-12 12:12:10"})
    snapshotter = SerizableObjectSnapshotter(fl_client)
    ckpt['reports_manager'] = snapshotter.save("reports_manager", ReportsManager)
    old_reports_manager = copy.deepcopy(fl_client.reports_manager)
    fl_client.reports_manager.report({"shutdown": "2012-12-12 12:12:12"})
    assert old_reports_manager.reporters[0].metrics != fl_client.reports_manager.reporters[0].metrics

    snapshotter.load(ckpt, "reports_manager", ReportsManager)
    assert old_reports_manager.reporters[0].metrics == fl_client.reports_manager.reporters[0].metrics

## LEFT OFF HERE
# def test_metric_manager_snapshotter() -> None:
#     metrics: Dict[str, Scalar] = {"test_metric": 1234}
#     reporter = JsonReporter()
#     fl_client = MockBasicClient(metrics=metrics,reporters=[reporter])
#     ckpt = {}

#     fl_client.reports_manager.report({"start": "2012-12-12 12:12:10"})
#     snapshotter = SerizableObjectSnapshotter(fl_client)
#     ckpt['reports_manager'] = snapshotter.save("reports_manager", ReportsManager)
#     old_reports_manager = copy.deepcopy(fl_client.reports_manager)
#     fl_client.reports_manager.report({"shutdown": "2012-12-12 12:12:12"})
#     assert old_reports_manager.reporters[0].metrics != fl_client.reports_manager.reporters[0].metrics

#     snapshotter.load(ckpt, "reports_manager", ReportsManager)
#     assert old_reports_manager.reporters[0].metrics == fl_client.reports_manager.reporters[0].metrics


class MockBasicClient(BasicClient):
    def __init__(
        self,
        loss_dict: Optional[Dict[str, float]] = None,
        metrics: Optional[Dict[str, Scalar]] = None,
        test_set_metrics: Optional[Dict[str, Scalar]] = None,
        loss: Optional[float] = 0,
        reporters: Sequence[BaseReporter] | None = None,
    ):
        super().__init__(Path(""), [], torch.device(0), reporters=reporters)

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
        self.max_num_validation_steps = None

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
        self.get_data_loaders.return_value = mock_data_loader, mock_data_loader
        self.get_test_data_loader = MagicMock()  # type: ignore
        self.get_test_data_loader.return_value = mock_data_loader
        self.get_optimizer = MagicMock()  # type: ignore
        self.get_criterion = MagicMock()  # type: ignore

        self._validate_or_test = MagicMock()  # type: ignore
        self._validate_or_test.side_effect = self.mock_validate_or_test

    def mock_validate_or_test(  # type: ignore
        self, loader, loss_meter, metric_manager, logging_mode=LoggingMode.VALIDATION, include_losses_in_metrics=False
    ):
        if include_losses_in_metrics:
            assert self.mock_loss_dict is not None and self.mock_metrics is not None
            fold_loss_dict_into_metrics(self.mock_metrics, self.mock_loss_dict, logging_mode)
        if logging_mode == LoggingMode.VALIDATION:
            return self.mock_loss, self.mock_metrics
        else:
            return self.mock_loss, self.mock_metrics_test