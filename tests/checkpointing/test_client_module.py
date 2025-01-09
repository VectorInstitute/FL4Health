from pathlib import Path

import pytest
import torch

from fl4health.checkpointing.checkpointer import (
    BestLossTorchModuleCheckpointer,
    LatestTorchModuleCheckpointer,
    TorchModuleCheckpointer,
)
from fl4health.checkpointing.client_module import CheckpointMode, ClientCheckpointAndStateModule
from fl4health.checkpointing.opacus_checkpointer import BestLossOpacusCheckpointer
from fl4health.utils.privacy_utilities import convert_model_to_opacus_model
from tests.test_utils.models_for_test import LinearTransform


def test_client_checkpointer_module_opacus(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path.joinpath("resources")
    checkpoint_dir.mkdir()
    pre_aggregation_checkpointer = BestLossOpacusCheckpointer(str(checkpoint_dir), "pre_agg.pkl")
    post_aggregation_checkpointer = BestLossOpacusCheckpointer(str(checkpoint_dir), "post_agg.pkl")
    checkpointer = ClientCheckpointAndStateModule(
        pre_aggregation=pre_aggregation_checkpointer, post_aggregation=post_aggregation_checkpointer
    )

    model_1 = LinearTransform()
    model_2 = LinearTransform()
    loaded_pre_model = LinearTransform()
    loaded_post_model = LinearTransform()

    model_1 = convert_model_to_opacus_model(model_1)
    model_2 = convert_model_to_opacus_model(model_2)

    checkpointer.maybe_checkpoint(model_1, 0.78, {"test_1": 1.0}, CheckpointMode.PRE_AGGREGATION)
    checkpointer.maybe_checkpoint(model_2, 0.78, {"test_1": 1.0}, CheckpointMode.POST_AGGREGATION)

    pre_aggregation_checkpointer.load_best_checkpoint_into_model(loaded_pre_model)
    post_aggregation_checkpointer.load_best_checkpoint_into_model(loaded_post_model)

    assert isinstance(loaded_pre_model, LinearTransform)
    # pre aggregation model should be the same as model_1
    assert torch.equal(model_1.linear.weight, loaded_pre_model.linear.weight)

    assert isinstance(loaded_post_model, LinearTransform)
    # post aggregation model should be the same as model_2
    assert torch.equal(model_2.linear.weight, loaded_post_model.linear.weight)

    checkpointer.maybe_checkpoint(model_2, 0.68, {"test_1": 1.0}, CheckpointMode.PRE_AGGREGATION)
    checkpointer.maybe_checkpoint(model_1, 0.68, {"test_1": 1.0}, CheckpointMode.POST_AGGREGATION)
    pre_aggregation_checkpointer.load_best_checkpoint_into_model(loaded_pre_model)
    post_aggregation_checkpointer.load_best_checkpoint_into_model(loaded_post_model)

    assert isinstance(loaded_pre_model, LinearTransform)
    # pre aggregation model should be the same as model_1
    assert torch.equal(model_2.linear.weight, loaded_pre_model.linear.weight)

    assert isinstance(loaded_post_model, LinearTransform)
    # post aggregation model should be the same as model_2
    assert torch.equal(model_1.linear.weight, loaded_post_model.linear.weight)


def test_client_checkpointer_module(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path.joinpath("resources")
    checkpoint_dir.mkdir()
    pre_aggregation_checkpointer = BestLossTorchModuleCheckpointer(str(checkpoint_dir), "pre_agg.pkl")
    post_aggregation_checkpointer = BestLossTorchModuleCheckpointer(str(checkpoint_dir), "post_agg.pkl")
    checkpointer = ClientCheckpointAndStateModule(
        pre_aggregation=pre_aggregation_checkpointer, post_aggregation=post_aggregation_checkpointer
    )

    model_1 = LinearTransform()
    model_2 = LinearTransform()

    checkpointer.maybe_checkpoint(model_1, 0.78, {"test_1": 1.0}, CheckpointMode.PRE_AGGREGATION)
    checkpointer.maybe_checkpoint(model_2, 0.78, {"test_1": 1.0}, CheckpointMode.POST_AGGREGATION)

    assert checkpointer.pre_aggregation is not None
    assert checkpointer.post_aggregation is not None
    loaded_pre_model = pre_aggregation_checkpointer.load_checkpoint()
    loaded_post_model = post_aggregation_checkpointer.load_checkpoint()

    assert isinstance(loaded_pre_model, LinearTransform)
    # pre aggregation model should be the same as model_1
    assert torch.equal(model_1.linear.weight, loaded_pre_model.linear.weight)

    assert isinstance(loaded_post_model, LinearTransform)
    # post aggregation model should be the same as model_2
    assert torch.equal(model_2.linear.weight, loaded_post_model.linear.weight)

    checkpointer.maybe_checkpoint(model_2, 0.68, {"test_1": 1.0}, CheckpointMode.PRE_AGGREGATION)
    checkpointer.maybe_checkpoint(model_1, 0.68, {"test_1": 1.0}, CheckpointMode.POST_AGGREGATION)
    loaded_pre_model = pre_aggregation_checkpointer.load_checkpoint()
    loaded_post_model = post_aggregation_checkpointer.load_checkpoint()

    assert isinstance(loaded_pre_model, LinearTransform)
    # pre aggregation model should be the same as model_1
    assert torch.equal(model_2.linear.weight, loaded_pre_model.linear.weight)

    assert isinstance(loaded_post_model, LinearTransform)
    # post aggregation model should be the same as model_2
    assert torch.equal(model_1.linear.weight, loaded_post_model.linear.weight)


def test_client_checkpointer_module_with_sequence_of_checkpointers(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path.joinpath("resources")
    checkpoint_dir.mkdir()
    pre_aggregation_checkpointer: list[TorchModuleCheckpointer] = [
        BestLossTorchModuleCheckpointer(str(checkpoint_dir), "pre_agg_best.pkl"),
        LatestTorchModuleCheckpointer(str(checkpoint_dir), "pre_agg_latest.pkl"),
    ]
    post_aggregation_checkpointer = BestLossTorchModuleCheckpointer(str(checkpoint_dir), "post_agg.pkl")
    checkpoint_module = ClientCheckpointAndStateModule(
        pre_aggregation=pre_aggregation_checkpointer, post_aggregation=post_aggregation_checkpointer
    )

    model_1 = LinearTransform()
    model_2 = LinearTransform()

    checkpoint_module.maybe_checkpoint(model_1, 0.78, {"test_1": 1.0}, CheckpointMode.PRE_AGGREGATION)
    checkpoint_module.maybe_checkpoint(model_2, 0.78, {"test_1": 1.0}, CheckpointMode.POST_AGGREGATION)

    assert checkpoint_module.pre_aggregation is not None
    assert checkpoint_module.post_aggregation is not None

    loaded_pre_model_best = pre_aggregation_checkpointer[0].load_checkpoint()
    loaded_pre_model_latest = pre_aggregation_checkpointer[1].load_checkpoint()
    loaded_post_model = post_aggregation_checkpointer.load_checkpoint()

    assert isinstance(loaded_pre_model_best, LinearTransform)
    # pre aggregation model should be the same as model_1
    assert torch.equal(model_1.linear.weight, loaded_pre_model_best.linear.weight)

    assert isinstance(loaded_pre_model_latest, LinearTransform)
    # pre aggregation model should be the same as model_1
    assert torch.equal(model_1.linear.weight, loaded_pre_model_latest.linear.weight)

    assert isinstance(loaded_post_model, LinearTransform)
    # post aggregation model should be the same as model_2
    assert torch.equal(model_2.linear.weight, loaded_post_model.linear.weight)

    checkpoint_module.maybe_checkpoint(model_2, 0.88, {"test_1": 1.0}, CheckpointMode.PRE_AGGREGATION)
    checkpoint_module.maybe_checkpoint(model_1, 0.68, {"test_1": 1.0}, CheckpointMode.POST_AGGREGATION)
    loaded_pre_model_best = pre_aggregation_checkpointer[0].load_checkpoint()
    loaded_pre_model_latest = pre_aggregation_checkpointer[1].load_checkpoint()
    loaded_post_model = post_aggregation_checkpointer.load_checkpoint()

    assert isinstance(loaded_pre_model_best, LinearTransform)
    # pre aggregation model should be the same as model_1 since the metric isn't better than the previous one
    assert torch.equal(model_1.linear.weight, loaded_pre_model_best.linear.weight)

    assert isinstance(loaded_pre_model_latest, LinearTransform)
    # pre aggregation model should be the same as model_2
    assert torch.equal(model_2.linear.weight, loaded_pre_model_latest.linear.weight)

    assert isinstance(loaded_post_model, LinearTransform)
    # post aggregation model should be the same as model_1
    assert torch.equal(model_1.linear.weight, loaded_post_model.linear.weight)


def test_path_duplication_check(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path.joinpath("resources")
    checkpoint_dir.mkdir()
    pre_aggregation_checkpointer = [
        BestLossTorchModuleCheckpointer(str(checkpoint_dir), "pre_agg_best.pkl"),
        LatestTorchModuleCheckpointer(str(checkpoint_dir), "pre_agg_best.pkl"),
    ]
    post_aggregation_checkpointer = BestLossTorchModuleCheckpointer(str(checkpoint_dir), "post_agg.pkl")
    # We have duplicate names, so we want to raise an error to prevent data loss/checkpoint overwrites
    with pytest.raises(ValueError):
        ClientCheckpointAndStateModule(
            pre_aggregation=pre_aggregation_checkpointer, post_aggregation=post_aggregation_checkpointer
        )

    pre_aggregation_checkpointer = [
        BestLossTorchModuleCheckpointer(str(checkpoint_dir), "pre_agg_best.pkl"),
        LatestTorchModuleCheckpointer(str(checkpoint_dir), "pre_agg_latest.pkl"),
    ]
    post_aggregation_checkpointer = BestLossTorchModuleCheckpointer(str(checkpoint_dir), "pre_agg_best.pkl")
    # We have duplicate names, so we want to raise an error to prevent data loss/checkpoint overwrites
    with pytest.raises(ValueError):
        ClientCheckpointAndStateModule(
            pre_aggregation=pre_aggregation_checkpointer, post_aggregation=post_aggregation_checkpointer
        )
