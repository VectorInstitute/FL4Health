import torch
from flwr.common.typing import Metrics
from pytest import LogCaptureFixture, approx

from fl4health.metrics.compound_metrics import EMAMetric
from fl4health.metrics.metrics import Accuracy
from fl4health.metrics.metrics_base import Metric


class DummyMetric(Metric):
    def __init__(self, name: str) -> None:
        self.acc = 0.0
        super().__init__(name)

    def update(self, input: torch.Tensor, target: torch.Tensor) -> None:
        pass

    def compute(self, name: str | None = None) -> Metrics:
        self.acc += 1.0
        return {"foo": "bar", "acc": self.acc}

    def clear(self) -> None:
        pass


def test_ema_metric_computation() -> None:
    preds_1 = torch.Tensor([0.9, 0.7, 0.6, 0.9, 0.1, 0.2, 0.99])
    targets_1 = torch.Tensor([1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0])

    preds_2 = torch.Tensor([0.19, 0.75, 0.26, 0.49, 0.12, 0.92, 0.99])
    targets_2 = torch.Tensor([1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0])

    preds_3 = torch.Tensor([0.19, 0.75, 0.26, 0.49])
    targets_3 = torch.Tensor([1.0, 0.0, 1.0, 0.0])

    accuracy_metric = Accuracy("accuracy")
    ema_accuracy_metric = EMAMetric(accuracy_metric, 0.2, name="ema_acc")

    # First accumulate two batches of updates before computation
    ema_accuracy_metric.update(preds_1, targets_1)
    ema_accuracy_metric.update(preds_2, targets_2)

    # Compute metric for first time
    metrics_1 = ema_accuracy_metric.compute()
    assert metrics_1["ema_acc"] == approx(6.0 / 14.0)

    # Now we clear the metric, as we are ready to start a new score for EMA
    ema_accuracy_metric.clear()

    # Compute the smoothed average between the previous metric and the current one
    ema_accuracy_metric.update(preds_3, targets_3)

    metrics_2 = ema_accuracy_metric.compute()
    assert metrics_2["ema_acc"] == approx(0.8 * (6.0 / 14.0) + 0.2 * (1.0 / 4.0), abs=1e-6)


def test_ema_warning_on_bad_type(caplog: LogCaptureFixture) -> None:

    dummy_metric = DummyMetric("dummy")
    ema_dummy_metric = EMAMetric(dummy_metric, 0.2, name="dummy")

    metrics_1 = ema_dummy_metric.compute("bad_ema")
    assert metrics_1["acc"] == 1.0

    assert "These values will be ignored in subsequent computations." in caplog.text

    metrics_2 = ema_dummy_metric.compute("bad_ema")
    assert metrics_2["acc"] == approx(0.8 * 1.0 + 0.2 * 2.0)
