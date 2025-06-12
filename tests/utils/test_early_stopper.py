from pathlib import Path
from unittest.mock import MagicMock

from fl4health.clients.basic_client import BasicClient
from fl4health.utils.early_stopper import EarlyStopper
from fl4health.utils.snapshotter import NumberSnapshotter


class MockBasicClient(BasicClient):

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


def test_early_stopper_patience_3(tmp_path: Path) -> None:
    mock_client = MockBasicClient()
    early_stopper = EarlyStopper(
        client=mock_client,
        train_loop_checkpoint_dir=tmp_path,
        patience=3,
        interval_steps=1,
    )
    # Override the snapshot_attrs of early stopper's state_checkpointer for test simplicity.
    early_stopper.state_checkpointer.snapshot_attrs = {
        "total_steps": (NumberSnapshotter(), int),
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
    mock_client = MockBasicClient()
    early_stopper = EarlyStopper(
        client=mock_client,
        train_loop_checkpoint_dir=tmp_path,
        patience=4,
        interval_steps=1,
    )
    # Override the snapshot_attrs of early stopper's state_checkpointer for test simplicity.
    early_stopper.state_checkpointer.snapshot_attrs = {
        "total_steps": (NumberSnapshotter(), int),
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
