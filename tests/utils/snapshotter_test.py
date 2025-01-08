import copy
from pathlib import Path

import torch

from fl4health.clients.basic_client import BasicClient
from fl4health.reporting import JsonReporter
from fl4health.reporting.reports_manager import ReportsManager
from fl4health.utils.losses import LossMeter, TrainingLosses
from fl4health.utils.metrics import Accuracy, MetricManager
from fl4health.utils.snapshotter import SerizableObjectSnapshotter
from fl4health.utils.typing import TorchPredType, TorchTargetType


def test_loss_meter_snapshotter() -> None:
    metrics = [Accuracy("accuracy")]
    reporter = JsonReporter()
    fl_client = BasicClient(data_path=Path(""), metrics=metrics, device=torch.device(0), reporters=[reporter])
    ckpt = {}

    fl_client.train_loss_meter.update(TrainingLosses(backward=torch.Tensor([35]), additional_losses=None))
    snapshotter = SerizableObjectSnapshotter(fl_client)
    ckpt["train_loss_meter"] = snapshotter.save("train_loss_meter", LossMeter)
    old_loss_meter = copy.deepcopy(fl_client.train_loss_meter)
    fl_client.train_loss_meter.update(TrainingLosses(backward=torch.Tensor([10]), additional_losses=None))
    assert len(old_loss_meter.losses_list) != len(fl_client.train_loss_meter.losses_list)

    snapshotter.load(ckpt, "train_loss_meter", LossMeter)

    assert len(old_loss_meter.losses_list) == len(fl_client.train_loss_meter.losses_list)
    for i in range(len(fl_client.train_loss_meter.losses_list)):
        assert old_loss_meter.losses_list[i].backward == fl_client.train_loss_meter.losses_list[i].backward
        assert (
            old_loss_meter.losses_list[i].additional_losses
            == fl_client.train_loss_meter.losses_list[i].additional_losses
        )


def test_reports_manager_snapshotter() -> None:
    metrics = [Accuracy("accuracy")]
    reporter = JsonReporter()
    fl_client = BasicClient(data_path=Path(""), metrics=metrics, device=torch.device(0), reporters=[reporter])
    ckpt = {}

    fl_client.reports_manager.report({"start": "2012-12-12 12:12:10"})
    snapshotter = SerizableObjectSnapshotter(fl_client)
    ckpt["reports_manager"] = snapshotter.save("reports_manager", ReportsManager)
    old_reports_manager = copy.deepcopy(fl_client.reports_manager)
    fl_client.reports_manager.report({"shutdown": "2012-12-12 12:12:12"})

    assert isinstance(old_reports_manager.reporters[0], JsonReporter) and isinstance(
        fl_client.reports_manager.reporters[0], JsonReporter
    )

    assert old_reports_manager.reporters[0].metrics != fl_client.reports_manager.reporters[0].metrics

    snapshotter.load(ckpt, "reports_manager", ReportsManager)
    assert old_reports_manager.reporters[0].metrics == fl_client.reports_manager.reporters[0].metrics


def test_metric_manager_snapshotter() -> None:
    metrics = [Accuracy("accuracy")]
    reporter = JsonReporter()
    fl_client = BasicClient(data_path=Path(""), metrics=metrics, device=torch.device(0), reporters=[reporter])
    ckpt = {}
    preds: TorchPredType = {
        "1": torch.tensor([0.7369, 0.5121, 0.2674, 0.5847, 0.4032, 0.7458, 0.9274, 0.3258, 0.7095, 0.0513])
    }
    target: TorchTargetType = {"1": torch.tensor([0, 1, 0, 1, 1, 0, 1, 1, 0, 1])}

    fl_client.train_metric_manager.update(preds, target)
    snapshotter = SerizableObjectSnapshotter(fl_client)
    ckpt["train_metric_manager"] = snapshotter.save("train_metric_manager", MetricManager)
    old_train_metric_manager = copy.deepcopy(fl_client.train_metric_manager)
    fl_client.train_metric_manager.update(preds, target)
    assert isinstance(fl_client.train_metric_manager.metrics_per_prediction_type["1"][0], Accuracy) and isinstance(
        old_train_metric_manager.metrics_per_prediction_type["1"][0], Accuracy
    )
    assert len(fl_client.train_metric_manager.metrics_per_prediction_type["1"][0].accumulated_inputs) != len(
        old_train_metric_manager.metrics_per_prediction_type["1"][0].accumulated_inputs
    )
    assert len(fl_client.train_metric_manager.metrics_per_prediction_type["1"][0].accumulated_targets) != len(
        old_train_metric_manager.metrics_per_prediction_type["1"][0].accumulated_targets
    )

    snapshotter.load(ckpt, "train_metric_manager", MetricManager)
    assert len(fl_client.train_metric_manager.metrics_per_prediction_type["1"][0].accumulated_inputs) == len(
        old_train_metric_manager.metrics_per_prediction_type["1"][0].accumulated_inputs
    )
    assert len(fl_client.train_metric_manager.metrics_per_prediction_type["1"][0].accumulated_targets) == len(
        old_train_metric_manager.metrics_per_prediction_type["1"][0].accumulated_targets
    )

    for i in range(len(fl_client.train_metric_manager.metrics_per_prediction_type["1"][0].accumulated_inputs)):
        assert torch.all(
            fl_client.train_metric_manager.metrics_per_prediction_type["1"][0].accumulated_inputs[i]
            == old_train_metric_manager.metrics_per_prediction_type["1"][0].accumulated_inputs[i]
        )
        assert torch.all(
            fl_client.train_metric_manager.metrics_per_prediction_type["1"][0].accumulated_targets[i]
            == old_train_metric_manager.metrics_per_prediction_type["1"][0].accumulated_targets[i]
        )
