from pathlib import Path

import pytest
import torch
from flwr.common.typing import Scalar

from fl4health.checkpointing.checkpointer import BestLossTorchModuleCheckpointer
from fl4health.checkpointing.opacus_checkpointer import (
    BestLossOpacusCheckpointer,
    LatestOpacusCheckpointer,
    OpacusCheckpointer,
)
from fl4health.utils.privacy_utilities import convert_model_to_opacus_model
from tests.checkpointing.utils import create_opacus_model_via_functorch
from tests.test_utils.models_for_test import LinearTransform


def test_save_and_load_best_loss_checkpoint(tmp_path: Path) -> None:
    # Temporary path to write pkl to, will be cleaned up at the end of the test.
    checkpoint_dir = tmp_path.joinpath("resources")
    checkpoint_dir.mkdir()

    model_1 = LinearTransform()
    model_2 = LinearTransform()
    target_model = LinearTransform()

    model_1 = convert_model_to_opacus_model(model_1)
    model_2 = convert_model_to_opacus_model(model_2)

    checkpointer = BestLossOpacusCheckpointer(str(checkpoint_dir), "best_model_state.pkl")
    checkpointer.maybe_checkpoint(model_1, 1.23, {"test": 1.2})
    checkpointer.maybe_checkpoint(model_2, 0.98, {"test": 1.2})

    # Correct metric saved.
    assert checkpointer.best_score == 0.98

    # Should throw a not implemented error
    with pytest.raises(NotImplementedError):
        _ = checkpointer.load_checkpoint()

    checkpointer.load_best_checkpoint_into_model(target_model)

    assert isinstance(target_model, LinearTransform)
    # Correct loading tensors of the second model with better loss value
    assert torch.equal(model_2.linear.weight, target_model.linear.weight)


def test_save_and_load_latest_checkpoint(tmp_path: Path) -> None:
    # Temporary path to write pkl to, will be cleaned up at the end of the test.
    checkpoint_dir = tmp_path.joinpath("resources")
    checkpoint_dir.mkdir()

    model_1 = LinearTransform()
    model_2 = LinearTransform()
    target_model = LinearTransform()

    model_1 = convert_model_to_opacus_model(model_1)
    model_2 = convert_model_to_opacus_model(model_2)

    checkpointer = LatestOpacusCheckpointer(str(checkpoint_dir), "best_model_state.pkl")
    checkpointer.maybe_checkpoint(model_2, 0.7, {"test": 1.2})
    checkpointer.maybe_checkpoint(model_1, 0.6, {"test": 1.2})

    # Should throw a not implemented error
    with pytest.raises(NotImplementedError):
        _ = checkpointer.load_checkpoint()

    checkpointer.load_best_checkpoint_into_model(target_model)
    assert isinstance(target_model, LinearTransform)
    # Correct loading tensors of the first model since each should be saved and model_1 is the "latest"
    assert torch.equal(model_1.linear.weight, target_model.linear.weight)


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
    checkpoint_name = "best_model_state.pkl"

    model_1 = LinearTransform()
    model_2 = LinearTransform()
    target_model = LinearTransform()

    model_1 = convert_model_to_opacus_model(model_1)
    model_2 = convert_model_to_opacus_model(model_2)

    opacus_checkpointer = OpacusCheckpointer(str(checkpoint_dir), checkpoint_name, score_function, maximize=True)
    loss_1, loss_2 = 1.0, 0.9
    metrics_1: dict[str, Scalar] = {"accuracy": 0.87, "precision": 0.67, "f1": 0.76}
    metrics_2: dict[str, Scalar] = {"accuracy": 0.87, "precision": 0.90, "f1": 0.60}
    opacus_checkpointer.best_score = 0.85

    # model_1 should not be checkpointed because the model score is lower than the best score set above
    # So the file should not exist
    opacus_checkpointer.maybe_checkpoint(model_1, loss_1, metrics_1)
    assert not checkpoint_dir.joinpath(checkpoint_name).is_file()

    # Should be true since the average of accuracy and precision provided in the dictionary is larger than 0.85
    opacus_checkpointer.maybe_checkpoint(model_2, loss_2, metrics_2)

    # Should throw a not implemented error
    with pytest.raises(NotImplementedError):
        _ = opacus_checkpointer.load_checkpoint()

    opacus_checkpointer.load_best_checkpoint_into_model(target_model)
    assert isinstance(target_model, LinearTransform)
    # Correct loading tensors of the first model since each should be saved and model_1 is the "latest"
    assert torch.equal(model_2.linear.weight, target_model.linear.weight)


def test_violation_of_opacus_object() -> None:
    # Placeholders as we won't be writing anything
    checkpoint_dir = "placeholder"
    checkpoint_name = "placeholder.pkl"

    model = LinearTransform()
    opacus_checkpointer = OpacusCheckpointer(checkpoint_dir, checkpoint_name, score_function, maximize=True)
    # Should throw an assertion error because the model is not an Opacus Grad Sampler to push people to use
    # a standard checkpointer.
    with pytest.raises(AssertionError):
        opacus_checkpointer.maybe_checkpoint(model, 0.45, {})


def test_fix_of_loss_stateless_model_exception(tmp_path: Path) -> None:
    # Temporary path to write pkl to, will be cleaned up at the end of the test.
    checkpoint_dir = tmp_path.joinpath("resources")
    checkpoint_dir.mkdir()
    checkpoint_name = "best_model_state.pkl"

    model = LinearTransform()
    target_model = LinearTransform()
    opacus_target_model = LinearTransform()

    model = create_opacus_model_via_functorch(model)
    opacus_target_model = convert_model_to_opacus_model(opacus_target_model)

    torch_checkpointer = BestLossTorchModuleCheckpointer(str(checkpoint_dir), checkpoint_name)
    # This should throw an error along the lines of
    # AttributeError: Can't pickle local object 'vmap.<locals>.wrapped'
    with pytest.raises(AttributeError) as attribute_exception:
        torch_checkpointer.maybe_checkpoint(model, 0.54, {})

    assert (
        str(attribute_exception.value)
        == "Can't pickle local object 'prepare_layer.<locals>.compute_loss_stateless_model'"
    )

    # The OpacusCheckpointer should suffer no such attribute error
    checkpointer = BestLossOpacusCheckpointer(str(checkpoint_dir), "best_model_state.pkl")
    checkpointer.maybe_checkpoint(model, 0.54, {"test": 1.2})

    checkpointer.load_best_checkpoint_into_model(target_model)

    assert isinstance(target_model, LinearTransform)
    # Verify correct loading tensors
    assert torch.equal(model.linear.weight, target_model.linear.weight)

    # Make sure that we can also load in the checkpoint into an Opacus module as well
    checkpointer.load_best_checkpoint_into_model(opacus_target_model, target_is_grad_sample_module=True)
    # Verify correct loading tensors
    assert torch.equal(model.linear.weight, opacus_target_model._module.linear.weight)
