from pathlib import Path

import torch
from flwr.common.typing import Scalar

from fl4health.checkpointing.checkpointer import (
    BestLossTorchModuleCheckpointer,
    FunctionTorchModuleCheckpointer,
    LatestTorchModuleCheckpointer,
)
from tests.test_utils.models_for_test import LinearTransform


def test_save_and_load_best_loss_checkpoint(tmp_path: Path) -> None:
    # Temporary path to write pkl to, will be cleaned up at the end of the test.
    checkpoint_dir = tmp_path.joinpath("resources")
    checkpoint_dir.mkdir()

    model_1 = LinearTransform()
    model_2 = LinearTransform()
    checkpointer = BestLossTorchModuleCheckpointer(str(checkpoint_dir), "best_model.pkl")
    checkpointer.maybe_checkpoint(model_1, 1.23, {"test": 1.2})
    checkpointer.maybe_checkpoint(model_2, 0.98, {"test": 1.2})

    # Correct metric saved.
    assert checkpointer.best_score == 0.98

    loaded_model = checkpointer.load_checkpoint()
    assert isinstance(loaded_model, LinearTransform)
    # Correct loading tensors of the second model with better loss value
    assert torch.equal(model_2.linear.weight, loaded_model.linear.weight)


def test_save_and_load_latest_checkpoint(tmp_path: Path) -> None:
    # Temporary path to write pkl to, will be cleaned up at the end of the test.
    checkpoint_dir = tmp_path.joinpath("resources")
    checkpoint_dir.mkdir()

    model_1 = LinearTransform()
    model_2 = LinearTransform()
    checkpointer = LatestTorchModuleCheckpointer(str(checkpoint_dir), "best_model.pkl")
    checkpointer.maybe_checkpoint(model_2, 0.7, {"test": 1.2})
    checkpointer.maybe_checkpoint(model_1, 0.6, {"test": 1.2})

    loaded_model = checkpointer.load_checkpoint()
    assert isinstance(loaded_model, LinearTransform)
    # Correct loading tensors of the first model since each should be saved and model_1 is the "latest"
    assert torch.equal(model_1.linear.weight, loaded_model.linear.weight)


def score_function(loss: float, metrics: dict[str, Scalar]) -> float:
    accuracy = metrics["accuracy"]
    precision = metrics["precision"]
    assert isinstance(accuracy, float)
    assert isinstance(precision, float)
    return 0.5 * (accuracy + precision)


def test_save_and_load_function_checkpoint(tmp_path: Path) -> None:
    # Temporary path to write pkl to, will be cleaned up at the end of the test.
    checkpoint_dir = tmp_path.joinpath("resources")
    checkpoint_dir.mkdir()
    checkpoint_name = "best_model.pkl"

    model_1 = LinearTransform()
    model_2 = LinearTransform()

    function_checkpointer = FunctionTorchModuleCheckpointer(
        str(checkpoint_dir), checkpoint_name, score_function, maximize=True
    )
    loss_1, loss_2 = 1.0, 0.9
    metrics_1: dict[str, Scalar] = {"accuracy": 0.87, "precision": 0.67, "f1": 0.76}
    metrics_2: dict[str, Scalar] = {"accuracy": 0.87, "precision": 0.90, "f1": 0.60}
    function_checkpointer.best_score = 0.85

    # model_1 should not be checkpointed because the model score is lower than the best score set above
    # So the file should not exist
    function_checkpointer.maybe_checkpoint(model_1, loss_1, metrics_1)
    assert not checkpoint_dir.joinpath(checkpoint_name).is_file()

    # Should be true since the average of accuracy and precision provided in the dictionary is larger than 0.85
    function_checkpointer.maybe_checkpoint(model_2, loss_2, metrics_2)
    loaded_model = function_checkpointer.load_checkpoint()
    assert isinstance(loaded_model, LinearTransform)
    # Correct loading tensors of the first model since each should be saved and model_1 is the "latest"
    assert torch.equal(model_2.linear.weight, loaded_model.linear.weight)
