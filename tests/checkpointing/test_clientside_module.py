from pathlib import Path

import torch

from fl4health.checkpointing.checkpointer import BestLossTorchCheckpointer
from fl4health.checkpointing.client_module import CheckpointMode, ClientCheckpointModule
from tests.test_utils.models_for_test import LinearTransform


def test_client_side_module(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path.joinpath("resources")
    checkpoint_dir.mkdir()
    pre_aggregation_checkpointer = BestLossTorchCheckpointer(str(checkpoint_dir), "pre_agg.pkl")
    post_aggregation_checkpointer = BestLossTorchCheckpointer(str(checkpoint_dir), "post_agg.pkl")
    checkpointer = ClientCheckpointModule(
        pre_aggregation=pre_aggregation_checkpointer, post_aggregation=post_aggregation_checkpointer
    )

    model_1 = LinearTransform()
    model_2 = LinearTransform()

    checkpointer.maybe_checkpoint(model_1, 0.78, {"test_1": 1.0}, CheckpointMode.PRE_AGGREGATION)
    checkpointer.maybe_checkpoint(model_2, 0.78, {"test_1": 1.0}, CheckpointMode.POST_AGGREGATION)

    assert checkpointer.pre_aggregation is not None
    assert checkpointer.post_aggregation is not None
    loaded_pre_model = checkpointer.pre_aggregation.load_best_checkpoint()
    loaded_post_model = checkpointer.post_aggregation.load_best_checkpoint()

    assert isinstance(loaded_pre_model, LinearTransform)
    # pre aggregation model should be the same as model_1
    assert torch.equal(model_1.linear.weight, loaded_pre_model.linear.weight)

    assert isinstance(loaded_post_model, LinearTransform)
    # post aggregation model should be the same as model_2
    assert torch.equal(model_2.linear.weight, loaded_post_model.linear.weight)

    checkpointer.maybe_checkpoint(model_2, 0.68, {"test_1": 1.0}, CheckpointMode.PRE_AGGREGATION)
    checkpointer.maybe_checkpoint(model_1, 0.68, {"test_1": 1.0}, CheckpointMode.POST_AGGREGATION)
    loaded_pre_model = checkpointer.pre_aggregation.load_best_checkpoint()
    loaded_post_model = checkpointer.post_aggregation.load_best_checkpoint()

    assert isinstance(loaded_pre_model, LinearTransform)
    # pre aggregation model should be the same as model_1
    assert torch.equal(model_2.linear.weight, loaded_pre_model.linear.weight)

    assert isinstance(loaded_post_model, LinearTransform)
    # post aggregation model should be the same as model_2
    assert torch.equal(model_1.linear.weight, loaded_post_model.linear.weight)
