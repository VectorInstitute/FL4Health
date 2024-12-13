import os
import tempfile
from pathlib import Path

import torch
from torch.optim import Optimizer

from fl4health.checkpointing.checkpointer import PerRoundStateCheckpointer
from tests.test_utils.models_for_test import LinearModel


def test_per_round_checkpointer() -> None:
    model: torch.nn.Module = LinearModel()
    optimizer: Optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    with tempfile.TemporaryDirectory() as results_dir:
        checkpoint_name = "ckpt.pt"
        checkpoint_path = os.path.join(results_dir, checkpoint_name)
        checkpointer = PerRoundStateCheckpointer(checkpoint_dir=Path(results_dir))

        assert not checkpointer.checkpoint_exists(checkpoint_path)

        checkpointer.save_checkpoint(
            checkpoint_name=checkpoint_name,
            checkpoint_dict={"model": model, "optimizer": optimizer, "current_round": 0},
        )

        assert checkpointer.checkpoint_exists(checkpoint_path)

        ckpt = checkpointer.load_checkpoint(checkpoint_path)

        assert "model" in ckpt and isinstance(ckpt["model"], torch.nn.Module)
        assert "optimizer" in ckpt and isinstance(ckpt["optimizer"], torch.optim.Optimizer)
        assert "current_round" in ckpt and isinstance(ckpt["current_round"], int)
