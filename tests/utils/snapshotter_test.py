import copy
from pathlib import Path

import torch

from fl4health.clients.basic_client import BasicClient
from fl4health.reporting import JsonReporter
from fl4health.reporting.reports_manager import ReportsManager
from fl4health.utils.losses import LossMeter, TrainingLosses
from fl4health.utils.metrics import Accuracy, MetricManager
from fl4health.utils.snapshotter import (
    LRSchedulerSnapshotter,
    NumberSnapshotter,
    OptimizerSnapshotter,
    SerizableObjectSnapshotter,
    TorchModuleSnapshotter,
)
from fl4health.utils.typing import TorchPredType, TorchTargetType
from tests.test_utils.models_for_test import SingleLayerWithSeed


def test_number_snapshotter() -> None:
    metrics = [Accuracy("accuracy")]
    reporter = JsonReporter()
    fl_client = BasicClient(data_path=Path(""), metrics=metrics, device=torch.device(0), reporters=[reporter])
    old_total_steps = fl_client.total_steps
    number_snapshotter = NumberSnapshotter(fl_client)
    sp = number_snapshotter.save("total_steps", int)
    fl_client.total_steps += 1
    assert sp["total_steps"] == {"None": old_total_steps}
    assert fl_client.total_steps != old_total_steps
    number_snapshotter.load(sp, "total_steps", int)
    assert fl_client.total_steps == old_total_steps


def test_optimizer_scheduler_model_snapshotter() -> None:
    metrics = [Accuracy("accuracy")]
    reporter = JsonReporter()
    fl_client = BasicClient(data_path=Path(""), metrics=metrics, device=torch.device(0), reporters=[reporter])
    fl_client.model = SingleLayerWithSeed()
    fl_client.criterion = torch.nn.CrossEntropyLoss()

    input_data = torch.randn(32, 100)  # Batch size = 32, Input size = 10
    target_data = torch.randn(32, 2)  # Batch size = 32, Target size = 1

    fl_client.optimizers = {"global": torch.optim.SGD(fl_client.model.parameters(), lr=0.001)}
    fl_client.lr_schedulers = {
        "global": torch.optim.lr_scheduler.StepLR(fl_client.optimizers["global"], step_size=30, gamma=0.1)
    }
    old_optimizers = copy.deepcopy(fl_client.optimizers)
    old_lr_schedulers = copy.deepcopy(fl_client.lr_schedulers)
    old_model = copy.deepcopy(fl_client.model)

    optimizer_snapshotter = OptimizerSnapshotter(fl_client)
    lr_scheduler_snapshotter = LRSchedulerSnapshotter(fl_client)
    model_snapshotter = TorchModuleSnapshotter(fl_client)

    snapshots = {}
    snapshots.update(optimizer_snapshotter.save("optimizers", torch.optim.Optimizer))
    snapshots.update(lr_scheduler_snapshotter.save("lr_schedulers", torch.optim.lr_scheduler.LRScheduler))
    snapshots.update(model_snapshotter.save("model", torch.nn.Module))

    fl_client.train_step(input_data, target_data)

    fl_client.optimizers["global"].step()  # Update model weights
    fl_client.lr_schedulers["global"].step()

    for key, value in fl_client.model.state_dict().items():
        assert not torch.equal(value, old_model.state_dict()[key])

    for key, optimizers in fl_client.optimizers.items():
        assert optimizers.state_dict()["state"] != old_optimizers[key].state_dict()["state"]

    for key, schedulers in fl_client.lr_schedulers.items():
        assert schedulers.state_dict() != old_lr_schedulers[key].state_dict()

    optimizer_snapshotter.load(snapshots, "optimizers", torch.optim.Optimizer)
    lr_scheduler_snapshotter.load(snapshots, "lr_schedulers", torch.optim.lr_scheduler.LRScheduler)
    model_snapshotter.load(snapshots, "model", torch.nn.Module)

    for key, value in fl_client.model.state_dict().items():
        assert torch.equal(value, old_model.state_dict()[key])

    for key, optimizers in fl_client.optimizers.items():
        assert optimizers.state_dict()["state"] == old_optimizers[key].state_dict()["state"]

    for key, schedulers in fl_client.lr_schedulers.items():
        assert schedulers.state_dict() == old_lr_schedulers[key].state_dict()


def test_loss_meter_snapshotter() -> None:
    metrics = [Accuracy("accuracy")]
    reporter = JsonReporter()
    fl_client = BasicClient(data_path=Path(""), metrics=metrics, device=torch.device(0), reporters=[reporter])
    snapshots = {}

    fl_client.train_loss_meter.update(TrainingLosses(backward=torch.Tensor([35]), additional_losses=None))
    snapshotter = SerizableObjectSnapshotter(fl_client)
    snapshots.update(snapshotter.save("train_loss_meter", LossMeter))
    old_loss_meter = copy.deepcopy(fl_client.train_loss_meter)
    fl_client.train_loss_meter.update(TrainingLosses(backward=torch.Tensor([10]), additional_losses=None))
    assert len(old_loss_meter.losses_list) != len(fl_client.train_loss_meter.losses_list)

    snapshotter.load(snapshots, "train_loss_meter", LossMeter)

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
    snapshots = {}

    fl_client.reports_manager.report({"start": "2012-12-12 12:12:10"})
    snapshotter = SerizableObjectSnapshotter(fl_client)
    snapshots.update(snapshotter.save("reports_manager", ReportsManager))
    old_reports_manager = copy.deepcopy(fl_client.reports_manager)
    fl_client.reports_manager.report({"shutdown": "2012-12-12 12:12:12"})

    assert isinstance(old_reports_manager.reporters[0], JsonReporter) and isinstance(
        fl_client.reports_manager.reporters[0], JsonReporter
    )

    assert old_reports_manager.reporters[0].metrics != fl_client.reports_manager.reporters[0].metrics

    snapshotter.load(snapshots, "reports_manager", ReportsManager)
    assert old_reports_manager.reporters[0].metrics == fl_client.reports_manager.reporters[0].metrics


def test_metric_manager_snapshotter() -> None:
    metrics = [Accuracy("accuracy")]
    reporter = JsonReporter()
    fl_client = BasicClient(data_path=Path(""), metrics=metrics, device=torch.device(0), reporters=[reporter])
    snapshots = {}
    preds: TorchPredType = {
        "1": torch.tensor([0.7369, 0.5121, 0.2674, 0.5847, 0.4032, 0.7458, 0.9274, 0.3258, 0.7095, 0.0513])
    }
    target: TorchTargetType = {"1": torch.tensor([0, 1, 0, 1, 1, 0, 1, 1, 0, 1])}

    fl_client.train_metric_manager.update(preds, target)
    snapshotter = SerizableObjectSnapshotter(fl_client)
    snapshots.update(snapshotter.save("train_metric_manager", MetricManager))
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

    snapshotter.load(snapshots, "train_metric_manager", MetricManager)
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
