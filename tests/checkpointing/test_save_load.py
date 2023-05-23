import os
from pathlib import Path

import torch
import torch.nn as nn

from fl4health.checkpointing.checkpointer import BestMetricTorchCheckpointer


class LinearTransform(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 3, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def test_save_and_load_checkpoint() -> None:
    checkpoint_dir = f"{Path(__file__). parent}/resources/"
    model_1 = LinearTransform()
    model_2 = LinearTransform()
    checkpointer = BestMetricTorchCheckpointer(checkpoint_dir, "best_model.pkl", True)
    checkpointer.maybe_checkpoint(model_1, 0.6)
    checkpointer.maybe_checkpoint(model_2, 0.7)

    # Correct metric saved.
    assert checkpointer.best_metric == 0.7

    loaded_model = checkpointer.load_best_checkpoint()
    assert isinstance(loaded_model, LinearTransform)
    # Correct loading tensors of the second model with better metric
    assert torch.equal(model_2.linear.weight, loaded_model.linear.weight)

    # clean up model artifact after passing
    os.remove(checkpointer.best_checkpoint_path)
