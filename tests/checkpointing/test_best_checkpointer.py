from pathlib import Path

import pytest
import torch
from flwr.common.typing import Scalar

from fl4health.checkpointing.checkpointer import (
    BestLossTorchModuleCheckpointer,
    BestMetricTorchModuleCheckpointer,
    FunctionTorchModuleCheckpointer,
)
from tests.test_utils.models_for_test import LinearTransform


def test_best_loss_checkpointer(tmp_path: Path) -> None:
    # Temporary path to write pkl to, will be cleaned up at the end of the test.
    checkpoint_dir = tmp_path.joinpath("resources")
    checkpoint_dir.mkdir()

    best_loss_checkpointer = BestLossTorchModuleCheckpointer(str(checkpoint_dir), "best_model.pkl")
    # First checkpoint should happen since the best loss is None
    none_checkpoint = best_loss_checkpointer._should_checkpoint(0.95)
    assert none_checkpoint
    best_loss_checkpointer.best_score = 0.95

    # Second checkpoint shouldn't happen since it's not smaller
    larger_loss_checkpoint = best_loss_checkpointer._should_checkpoint(1.1)
    assert not larger_loss_checkpoint

    # third checkpoint should happen as it is smaller
    smaller_loss_checkpoint = best_loss_checkpointer._should_checkpoint(0.87)
    assert smaller_loss_checkpoint

    # Test the actual checkpointing part, should checkpoint as the loss is smallest
    linear_1 = LinearTransform()
    best_loss_checkpointer.maybe_checkpoint(linear_1, 0.85, metrics={})

    loaded_model = best_loss_checkpointer.load_checkpoint()
    assert isinstance(loaded_model, LinearTransform)
    # Correctly loading tensors of the model with better loss value
    assert torch.equal(linear_1.linear.weight, loaded_model.linear.weight)


def test_best_metric_checkpointer(tmp_path: Path) -> None:
    # Temporary path to write pkl to, will be cleaned up at the end of the test.
    checkpoint_dir = tmp_path.joinpath("resources")
    checkpoint_dir.mkdir()

    linear_1 = LinearTransform()
    linear_2 = LinearTransform()

    best_metric_checkpointer = BestMetricTorchModuleCheckpointer(
        str(checkpoint_dir), "best_model.pkl", metric="dummy_metric", prefix="val - prefix - ", maximize=True
    )
    # First checkpoint should happen since the best metric is None
    none_checkpoint = best_metric_checkpointer._should_checkpoint(0.72)
    assert none_checkpoint
    best_metric_checkpointer.best_score = 0.72

    # first real checkpoint should happen since it's a better value
    best_metric_checkpointer.maybe_checkpoint(linear_1, -1.0, metrics={"val - prefix - dummy_metric": 0.78})

    loaded_model = best_metric_checkpointer.load_checkpoint()
    assert isinstance(loaded_model, LinearTransform)
    # Correctly loading tensors of the model with better loss value
    assert torch.equal(linear_1.linear.weight, loaded_model.linear.weight)

    # Second checkpoint shouldn't happen since it's not larger
    best_metric_checkpointer.maybe_checkpoint(
        linear_2, 100, metrics={"val - prefix - dummy_metric": 0.48, "test": 0.85}
    )

    loaded_model = best_metric_checkpointer.load_checkpoint()
    assert isinstance(loaded_model, LinearTransform)
    # Correctly loading tensors of the model with better loss value
    assert torch.equal(linear_1.linear.weight, loaded_model.linear.weight)

    # Now we want to make sure we throw an error if the metric key isn't in there
    with pytest.raises(KeyError):
        best_metric_checkpointer.maybe_checkpoint(linear_1, -1.0, metrics={"bad key": 0.78})

    # Test that it throws and error upon unsuccessful conversion of metric to float
    with pytest.raises(ValueError):
        best_metric_checkpointer.maybe_checkpoint(
            linear_1, -1.0, metrics={"val - prefix - dummy_metric": "bad metric value"}
        )


def test_default_naming_of_score() -> None:
    def score_func_with_name(_: float, metrics: dict[str, Scalar]) -> float:
        return 1.0

    test_checkpoint = FunctionTorchModuleCheckpointer("", "", score_func_with_name)
    assert test_checkpoint.checkpoint_score_name == "score_func_with_name"
