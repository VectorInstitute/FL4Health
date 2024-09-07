import tempfile
from pathlib import Path
from typing import Dict

import torch
from flwr.server.history import History
from torch.optim import Optimizer

from fl4health.checkpointing.checkpointer import (
    CentralPerRoundCheckpointer,
    ClientPerRoundCheckpointer,
    ServerPerRoundCheckpointer,
)
from tests.test_utils.models_for_test import LinearModel


def test_central_checkpointer() -> None:
    model: torch.nn.Module = LinearModel()
    optimizer: Optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    with tempfile.TemporaryDirectory() as results_dir:
        central_checkpointer = CentralPerRoundCheckpointer(
            checkpoint_dir=Path(results_dir), checkpoint_name=Path("ckpt.pt")
        )

        assert not central_checkpointer.checkpoint_exists()

        central_checkpointer.save_checkpoint({"model": model, "optimizer": optimizer, "epoch": 0})

        assert central_checkpointer.checkpoint_exists()

        model, optimizer, epoch = central_checkpointer.load_checkpoint()

        assert isinstance(model, torch.nn.Module)
        assert isinstance(optimizer, torch.optim.SGD)
        assert isinstance(epoch, int)


def test_server_checkpointer() -> None:
    model: torch.nn.Module = LinearModel()
    history = History()
    with tempfile.TemporaryDirectory() as results_dir:
        server_checkpointer = ServerPerRoundCheckpointer(
            checkpoint_dir=Path(results_dir), checkpoint_name=Path("ckpt.pt")
        )

        assert not server_checkpointer.checkpoint_exists()

        server_checkpointer.save_checkpoint({"model": model, "history": history, "server_round": 0})

        assert server_checkpointer.checkpoint_exists()

        model, history, server_round, _ = server_checkpointer.load_checkpoint()

        assert isinstance(model, torch.nn.Module)
        assert isinstance(history, History)
        assert isinstance(server_round, int)


def test_client_checkpointer() -> None:
    model: torch.nn.Module = LinearModel()
    optimizers: Dict[str, Optimizer] = {"optimizer": torch.optim.SGD(model.parameters(), lr=0.01)}
    with tempfile.TemporaryDirectory() as results_dir:
        client_checkpointer = ClientPerRoundCheckpointer(
            checkpoint_dir=Path(results_dir), checkpoint_name=Path("ckpt.pt")
        )

        assert not client_checkpointer.checkpoint_exists()

        client_checkpointer.save_checkpoint(
            {"model": model, "optimizers": optimizers, "client_name": "bob", "total_steps": 3, "lr_schedulers": {}}
        )

        assert client_checkpointer.checkpoint_exists()

        model, optimizers, client_name, total_steps, lr_schedulers, _ = client_checkpointer.load_checkpoint()

        assert isinstance(model, torch.nn.Module)
        assert isinstance(optimizers, dict)
        assert isinstance(client_name, str)
        assert isinstance(total_steps, int)
        assert isinstance(lr_schedulers, dict)
