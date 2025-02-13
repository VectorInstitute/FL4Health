import pytest
import torch
from flwr.common import Scalar
from torch.utils.data import DataLoader

from fl4health.utils.client import (
    clone_and_freeze_model,
    fold_loss_dict_into_metrics,
    process_and_check_validation_steps,
)
from fl4health.utils.dataset import TensorDataset
from fl4health.utils.logging import LoggingMode
from tests.test_utils.models_for_test import LinearModel


def get_dummy_dataset() -> TensorDataset:
    data = torch.randn(100, 10, 8)
    targets = torch.randint(5, (100,))
    return TensorDataset(data=data, targets=targets)


DUMMY_DATASET = get_dummy_dataset()


def test_process_and_check_validation_steps(caplog: pytest.LogCaptureFixture) -> None:

    dataloader = DataLoader(DUMMY_DATASET, batch_size=15, shuffle=False)

    # No entry in the config, stays None
    num_validation_steps = process_and_check_validation_steps({}, dataloader)
    assert num_validation_steps is None

    # Entry is valid and smaller than the dataloader
    num_validation_steps = process_and_check_validation_steps({"num_validation_steps": 4}, dataloader)

    # Entry is invalid at 0
    with pytest.raises(AssertionError):
        num_validation_steps = process_and_check_validation_steps({"num_validation_steps": 0}, dataloader)

    # Raise warning if too large
    num_validation_steps = process_and_check_validation_steps({"num_validation_steps": 200}, dataloader)
    assert "is larger than the length of the" in caplog.text


def test_fold_loss_dict_into_metrics() -> None:
    metrics: dict[str, Scalar] = {"metric_1": 1.2, "metric_2": 3.2}
    loss = {"loss_1": 1.23, "loss_2": 0.93}
    fold_loss_dict_into_metrics(metrics, loss, LoggingMode.VALIDATION)
    assert metrics["val - loss_1"] == 1.23
    assert metrics["val - loss_2"] == 0.93
    assert metrics["metric_1"] == 1.2
    assert metrics["metric_2"] == 3.2

    metrics = {"metric_1": 1.2, "metric_2": 3.2}
    fold_loss_dict_into_metrics(metrics, loss, LoggingMode.TEST)
    assert metrics["test - loss_1"] == 1.23
    assert metrics["test - loss_2"] == 0.93
    assert metrics["metric_1"] == 1.2
    assert metrics["metric_2"] == 3.2


def test_clone_and_freeze_model() -> None:
    linear_model = LinearModel()
    linear_model.train()
    cloned_model = clone_and_freeze_model(linear_model)

    assert not cloned_model.training
    for param in cloned_model.parameters():
        assert not param.requires_grad

    assert linear_model.training

    for param in linear_model.parameters():
        assert param.requires_grad
