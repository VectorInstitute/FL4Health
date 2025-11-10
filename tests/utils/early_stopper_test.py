from collections.abc import Sequence
from pathlib import Path
from unittest.mock import MagicMock

import torch
from flwr.common import Scalar

from fl4health.clients.basic_client import BasicClient
from fl4health.reporting.base_reporter import BaseReporter
from fl4health.utils.client import fold_loss_dict_into_metrics
from fl4health.utils.early_stopper import EarlyStopper
from fl4health.utils.logging import LoggingMode
from fl4health.utils.losses import EvaluationLosses
from fl4health.utils.snapshotter import SingletonSnapshotter
from fl4health.utils.typing import TorchInputType, TorchTargetType


DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MinimalMockBasicClient(BasicClient):
    def __init__(self):  # type: ignore
        # Val loss first goes down, then up (for less than interval steps), then down and up again
        val_loss_values = [
            (0.5, None),
            (0.4, None),
            (0.3, None),
            (0.1, None),
            (0.15, None),
            (0.2, None),
            (0.05, None),
            (0.15, None),
            (0.2, None),
            (0.3, None),
            (0.01, None),
        ]
        self._fully_validate_or_test = MagicMock(side_effect=iter(val_loss_values))  # type: ignore
        self.total_steps = 0  # Tracking the total training steps before early stopping. # type: ignore
        self.client_name = "mock_client"  # type: ignore
        self.val_loader = None  # type: ignore
        self.val_loss_meter = None  # type: ignore
        self.val_metric_manager = None  # type: ignore


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

        self.test_attribute_number = 10
        self.learning_rate = 0.0001

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
        self.val_loader = MagicMock()
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


def test_early_stopper_creation_and_should_stop(tmp_path: Path) -> None:
    snapshot_dir = tmp_path.joinpath("resources")
    snapshot_dir.mkdir()
    client = MockBasicClient()
    early_stopper = EarlyStopper(client=client, patience=2, interval_steps=2, train_loop_checkpoint_dir=snapshot_dir)

    early_stopper.state_checkpointer.add_to_snapshot_attr("test_attribute_number", SingletonSnapshotter(), int)
    # Delete out the complex objects
    early_stopper.state_checkpointer.delete_from_snapshot_attr("model")
    early_stopper.state_checkpointer.delete_from_snapshot_attr("optimizers")
    early_stopper.state_checkpointer.delete_from_snapshot_attr("lr_schedulers")
    early_stopper.state_checkpointer.delete_from_snapshot_attr("reports_manager")
    early_stopper.state_checkpointer.delete_from_snapshot_attr("train_loss_meter")
    early_stopper.state_checkpointer.delete_from_snapshot_attr("train_metric_manager")

    early_stopper.state_checkpointer.save_client_state(early_stopper.client)
    early_stopper.load_snapshot()

    # Should be false since it's not the right interval (i.e. step divisible by 2)
    assert not early_stopper.should_stop(1)

    # Since mock loss is none, we should return False
    client.mock_loss = None
    assert not early_stopper.should_stop(2)

    # We'll still return false since best_score is going to be None, but we verify that it is now set
    client.mock_loss = 1.2
    assert not early_stopper.should_stop(4)
    assert early_stopper.best_score == 1.2

    # Patience should be decremented but early stopping should not be triggered
    client.mock_loss = 1.5
    assert not early_stopper.should_stop(6)
    assert early_stopper.count_down == 1

    # Patience should be reset and best score should be new best loss
    client.mock_loss = 0.9
    # We change the attribute number so we can verify that it is saved and then properly loaded.
    client.test_attribute_number = 9
    assert not early_stopper.should_stop(8)
    assert early_stopper.count_down == 2
    assert early_stopper.best_score == 0.9

    # Patience should be decremented to 0 and we should get a stop signal
    client.mock_loss = 1.1
    client.test_attribute_number = 10
    assert not early_stopper.should_stop(10)
    assert early_stopper.should_stop(12)

    early_stopper.load_snapshot()

    assert client.test_attribute_number == 9


def test_early_stopper_patience_3(tmp_path: Path) -> None:
    mock_client = MinimalMockBasicClient()
    early_stopper = EarlyStopper(
        client=mock_client,
        train_loop_checkpoint_dir=tmp_path,
        patience=3,
        interval_steps=1,
    )
    # Override the snapshot_attrs of early stopper's state_checkpointer for test simplicity.
    early_stopper.state_checkpointer.snapshot_attrs = {
        "total_steps": (SingletonSnapshotter(), int),
    }

    # Simulate training loop
    for step in range(10):
        mock_client.total_steps += 1
        if early_stopper.should_stop(step):
            assert mock_client.total_steps == 10
            early_stopper.load_snapshot()
            break
    assert step == 9
    # Patience is 3, so we never get to see the actual best loss of 0.01
    assert early_stopper.best_score == 0.05
    # early stopper continues for 3 steps after the best score
    assert early_stopper.count_down == 0
    assert mock_client.total_steps == 7


def test_early_stopper_patience_4(tmp_path: Path) -> None:
    mock_client = MinimalMockBasicClient()
    early_stopper = EarlyStopper(
        client=mock_client,
        train_loop_checkpoint_dir=tmp_path,
        patience=4,
        interval_steps=1,
    )
    # Override the snapshot_attrs of early stopper's state_checkpointer for test simplicity.
    early_stopper.state_checkpointer.snapshot_attrs = {
        "total_steps": (SingletonSnapshotter(), int),
    }

    # Simulate training loop
    for step in range(11):
        mock_client.total_steps += 1
        if early_stopper.should_stop(step):
            assert mock_client.total_steps == 11
            early_stopper.load_snapshot()
            break
    assert step == 10
    # Patience is 4, so we get to see the actual best loss of 0.01
    assert early_stopper.best_score == 0.01
    # early stopper's count_down is just restored to the max value
    assert early_stopper.count_down == 4
    assert mock_client.total_steps == 11
