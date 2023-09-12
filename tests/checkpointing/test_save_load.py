from pathlib import Path

import torch

from fl4health.checkpointing.checkpointer import BestMetricTorchCheckpointer
from tests.test_utils.models_for_test import LinearTransform


def test_save_and_load_checkpoint(tmp_path: Path) -> None:
    # Temporary path to write pkl to, will be cleaned up at the end of the test.
    checkpoint_dir = tmp_path.joinpath("resources")
    checkpoint_dir.mkdir()

    model_1 = LinearTransform()
    model_2 = LinearTransform()
    checkpointer = BestMetricTorchCheckpointer(str(checkpoint_dir), "best_model.pkl", True)
    checkpointer.maybe_checkpoint(model_1, 0.6)
    checkpointer.maybe_checkpoint(model_2, 0.7)

    # Correct metric saved.
    assert checkpointer.best_metric == 0.7

    loaded_model = checkpointer.load_best_checkpoint()
    assert isinstance(loaded_model, LinearTransform)
    # Correct loading tensors of the second model with better metric
    assert torch.equal(model_2.linear.weight, loaded_model.linear.weight)
