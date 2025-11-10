import copy
import pickle
from pathlib import Path
from typing import Any

import torch
from flwr.server.client_manager import SimpleClientManager
from flwr.server.history import History
from torch.optim import Optimizer

from fl4health.checkpointing.state_checkpointer import (
    ClientStateCheckpointer,
    NnUnetServerStateCheckpointer,
    ServerStateCheckpointer,
)
from fl4health.clients.basic_client import BasicClient
from fl4health.reporting import JsonReporter
from fl4health.servers.base_server import FlServer
from fl4health.servers.nnunet_server import NnunetServer
from fl4health.utils.metrics import Accuracy
from fl4health.utils.snapshotter import (
    AbstractSnapshotter,
    OptimizerSnapshotter,
    SingletonSnapshotter,
    StringSnapshotter,
)
from tests.test_utils.models_for_test import SingleLayerWithSeed


DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_fl_client() -> BasicClient:
    metrics = [Accuracy("accuracy")]
    reporter = JsonReporter()
    fl_client = BasicClient(
        data_path=Path(""),
        metrics=metrics,
        device=DEVICE,
        reporters=[reporter],
        client_name="original_client",
    )
    fl_client.model = SingleLayerWithSeed(seed=42, output_size=1)
    fl_client.optimizers = {"global": torch.optim.SGD(fl_client.model.parameters(), lr=0.001, momentum=0.1)}
    fl_client.lr_schedulers = {
        "global": torch.optim.lr_scheduler.StepLR(fl_client.optimizers["global"], step_size=30, gamma=0.1)
    }
    fl_client.criterion = torch.nn.CrossEntropyLoss()
    return fl_client


def test_client_state_works_for_per_round_checkpointing(tmp_path: Path) -> None:
    fl_client = create_fl_client()
    # Temporary path to save the state to, will be cleaned up at the end of the test.
    checkpoint_dir = tmp_path.joinpath("client_state")
    checkpoint_dir.mkdir()
    # checkpoint_dir is required for ClientStateCheckpointer to assist saving the state to the disk.
    # This is useful for resuming training after an interruption.
    checkpointer = ClientStateCheckpointer(checkpoint_dir=Path(checkpoint_dir), checkpoint_name="client_state.pt")
    copy_client = copy.deepcopy(fl_client)

    assert not checkpointer.checkpoint_exists()

    # Train the client with random data for one step, this updates the model and the optimizer state.
    input_data = torch.randn(32, 100)
    target_data = torch.randn(32, 1)
    fl_client.train_step(input_data, target_data)
    # Update lr_schedulers
    fl_client.update_lr_schedulers(step=0)
    # Update the step count
    fl_client.total_steps += 1

    # Save the state of the trained client
    checkpointer.save_client_state(fl_client)

    # Reset the checkpointer object
    # This is similar to the situation where we have to restart FL after an interruption.
    checkpointer = ClientStateCheckpointer(checkpoint_dir=Path(checkpoint_dir), checkpoint_name="client_state.pt")

    # Checkpoint should now exist
    assert checkpointer.checkpoint_exists()

    snapshot_ckpt = checkpointer.load_checkpoint()
    assert "lr_schedulers" in snapshot_ckpt
    assert "optimizers" in snapshot_ckpt
    assert "total_steps" in snapshot_ckpt
    assert "total_epochs" in snapshot_ckpt
    assert "reports_manager" in snapshot_ckpt

    # Copy client should have a different model state than the original client
    assert copy_client.total_steps == 0
    default_optimizer_state = copy_client.optimizers["global"].state_dict()["state"]
    updated_optimizer_state = fl_client.optimizers["global"].state_dict()["state"]
    assert default_optimizer_state == {}
    assert updated_optimizer_state != default_optimizer_state
    assert copy_client.lr_schedulers["global"].state_dict() != fl_client.lr_schedulers["global"].state_dict()

    # Loads fl_client checkpoint (trained client) into the copy client
    checkpointer.maybe_load_client_state(copy_client)

    # Check that state is loaded correctly.
    # Check the state of the loaded optimizer.
    loaded_optimizer_state = copy_client.optimizers["global"].state_dict()["state"]
    saved_optimizer_state = fl_client.optimizers["global"].state_dict()["state"]
    assert all(
        torch.equal(loaded_optimizer_state[0][key], saved_optimizer_state[0][key]) for key in loaded_optimizer_state[0]
    )
    # Check the state of the loaded lr_scheduler.
    assert copy_client.lr_schedulers["global"].state_dict() == fl_client.lr_schedulers["global"].state_dict()
    # Check the loaded total steps.
    assert copy_client.total_steps == 1


def test_client_state_works_for_training_loop_checkpointing(tmp_path: Path) -> None:
    """
    Train loop checkpointing requires more information than per round as it needs to be able to restart a training
    process where it left off. This is important for early stopping.
    """
    fl_client = create_fl_client()
    # Create a copy of the client for later.
    copy_client = copy.deepcopy(fl_client)
    # Temporary path to save the state to, will be cleaned up at the end of the test.
    checkpoint_dir = tmp_path.joinpath("client_state")
    checkpoint_dir.mkdir()
    checkpointer = ClientStateCheckpointer(checkpoint_dir=Path(checkpoint_dir), checkpoint_name="client_state.pt")

    # Simulate updates inside the training loop (one step).
    input_data = torch.randn(32, 100)
    target_data = torch.zeros(32, 1)
    report_data: dict[str, Any] = {"round": 0}
    losses, preds = fl_client.train_step(input_data, target_data)
    fl_client.train_loss_meter.update(losses)
    fl_client.update_metric_manager(preds, target_data, fl_client.train_metric_manager)
    fl_client.update_lr_schedulers(step=0)
    report_data.update({"fit_step_losses": losses.as_dict(), "fit_step": 0})
    report_data.update(fl_client.get_client_specific_reports())
    fl_client.reports_manager.report(report_data, 0, None, 0)
    fl_client.total_steps += 1

    # Save the state of the client.
    checkpointer.save_client_state(fl_client)

    assert checkpointer.checkpoint_exists()
    snapshot_ckpt = checkpointer.load_checkpoint()
    # Check that all the attributes defined in snapshot_attrs are present in the checkpoint.
    assert "model" in snapshot_ckpt
    assert "lr_schedulers" in snapshot_ckpt
    assert "optimizers" in snapshot_ckpt
    assert "total_steps" in snapshot_ckpt
    assert "total_epochs" in snapshot_ckpt
    assert "reports_manager" in snapshot_ckpt
    assert "train_loss_meter" in snapshot_ckpt
    assert "train_metric_manager" in snapshot_ckpt

    # Use the copy_client to load the state.
    checkpointer.maybe_load_client_state(copy_client)

    # Check that the state is loaded correctly.
    # Check the state of the loaded optimizer.
    loaded_optimizer_state = copy_client.optimizers["global"].state_dict()["state"]
    saved_optimizer_state = fl_client.optimizers["global"].state_dict()["state"]
    assert all(
        torch.equal(loaded_optimizer_state[0][key], saved_optimizer_state[0][key]) for key in loaded_optimizer_state[0]
    )
    # Check the state of the loaded lr_scheduler.
    assert copy_client.lr_schedulers["global"].state_dict() == fl_client.lr_schedulers["global"].state_dict()
    # Check the loaded total steps.
    assert copy_client.total_steps == 1
    # Check the loaded train loss meter.
    assert all(
        loss1.as_dict() == loss2.as_dict()
        for loss1, loss2 in zip(copy_client.train_loss_meter.losses_list, fl_client.train_loss_meter.losses_list)
    )
    # Check the loaded train metric manager.
    print(fl_client.train_metric_manager.compute())
    assert copy_client.train_metric_manager.compute() == fl_client.train_metric_manager.compute()
    # Check the loaded model.
    for key, value in copy_client.model.state_dict().items():
        assert torch.equal(value, fl_client.model.state_dict()[key])
    # Check the loaded reports manager.
    assert isinstance(copy_client.reports_manager.reporters[0], JsonReporter)  # Added for type checking
    assert isinstance(fl_client.reports_manager.reporters[0], JsonReporter)  # Added for type checking
    assert copy_client.reports_manager.reporters[0].metrics == fl_client.reports_manager.reporters[0].metrics


def test_client_state_checkpointing_with_custom_attrs(tmp_path: Path) -> None:
    """
    Test that the client can save and load its state based a custom set of attributes rather than the
    default.
    """
    checkpoint_dir = tmp_path.joinpath("client_state")
    checkpoint_dir.mkdir()

    fl_client = create_fl_client()
    # Create a copy of the client for later.
    copy_client = copy.deepcopy(fl_client)

    snapshot_attrs: dict[str, tuple[AbstractSnapshotter[Any], Any]] = {
        "total_steps": (SingletonSnapshotter(), int),
        "optimizers": (OptimizerSnapshotter(), Optimizer),
        "client_name": (StringSnapshotter(), str),
    }

    checkpointer = ClientStateCheckpointer(checkpoint_dir, None, snapshot_attrs)  # ignore: typing

    # Perform one training step.
    input_data = torch.randn(32, 100)
    target_data = torch.zeros(32, 1)
    fl_client.train_step(input_data, target_data)
    fl_client.total_steps += 2
    # Change the name of the client for testing purposes.
    fl_client.client_name = "updated_client"

    checkpointer.save_client_state(fl_client)
    assert checkpointer.checkpoint_exists()

    # Use the copy_client to load the state.
    checkpointer.maybe_load_client_state(copy_client)

    assert "reports_manager" not in checkpointer.snapshot_ckpt

    # Check that the state is loaded correctly.
    # Check the state of the loaded optimizer.
    loaded_optimizer_state = copy_client.optimizers["global"].state_dict()["state"]
    saved_optimizer_state = fl_client.optimizers["global"].state_dict()["state"]
    assert all(
        torch.equal(loaded_optimizer_state[0][key], saved_optimizer_state[0][key]) for key in loaded_optimizer_state[0]
    )

    # Check the loaded total steps.
    assert copy_client.total_steps == 2
    # Check the loaded client name.
    assert copy_client.client_name == "updated_client"


def test_server_state_checkpointer(tmp_path: Path) -> None:
    """Test the server state checkpointer."""
    fl_server = FlServer(
        client_manager=SimpleClientManager(),
        fl_config={"": ""},
        reporters=[JsonReporter()],
        server_name="original_server",
    )
    fl_server_model = SingleLayerWithSeed(seed=42, output_size=1)
    fl_server.history = History()
    # Create some history.
    fl_server.history.add_loss_distributed(0, 0.6)
    fl_server.history.add_loss_distributed(1, 0.4)
    fl_server.history.add_loss_centralized(0, 0.3)
    fl_server.history.add_loss_centralized(1, 0.2)
    fl_server.reports_manager.report(
        {
            "fit_start": "10.10",
            "host_type": "server",
        }
    )
    fl_server.current_round = 1
    # Temporary path to save the state to, will be cleaned up at the end of the test.
    checkpoint_dir = tmp_path.joinpath("server_state")
    checkpoint_dir.mkdir()

    # Create a checkpointer for the server.
    checkpointer = ServerStateCheckpointer(checkpoint_dir=Path(checkpoint_dir), checkpoint_name="server_state.pt")

    # Check that the checkpoint does not exist initially.
    assert not checkpointer.checkpoint_exists()

    # Save the state with the model.
    checkpointer.save_server_state(server=fl_server, model=fl_server_model)

    # Check that the checkpoint now exists.
    assert checkpointer.checkpoint_exists()

    # Create a new server to load the state into.
    new_server = FlServer(client_manager=SimpleClientManager(), fl_config={"": ""}, server_name="new_server")
    new_server_model = SingleLayerWithSeed(seed=12, output_size=1)
    new_server.history = History()
    new_server.current_round = 0

    # Check that new server is different from the original server.
    assert not all(
        torch.equal(value, fl_server_model.state_dict()[key]) for key, value in new_server_model.state_dict().items()
    )
    assert new_server.current_round != fl_server.current_round
    assert not all(
        getattr(new_server.history, key) == getattr(fl_server.history, key) for key in vars(fl_server.history)
    )
    assert new_server.server_name != fl_server.server_name
    # No reporters are set up in the new server.
    assert len(new_server.reports_manager.reporters) == 0

    # Now load the state into the new server.
    # We don't need to create a new checkpointer here,
    # but we create one to simulate the situation where the program is
    # restarted due to an interruption.
    new_checkpointer = ServerStateCheckpointer(checkpoint_dir=Path(checkpoint_dir), checkpoint_name="server_state.pt")
    new_checkpointer.maybe_load_server_state(server=new_server, model=new_server_model)

    # Check that the state is loaded correctly.
    assert all(
        torch.equal(value, fl_server_model.state_dict()[key]) for key, value in new_server_model.state_dict().items()
    )
    assert new_server.current_round == fl_server.current_round
    # Check that the history is loaded correctly.
    assert all(getattr(new_server.history, key) == getattr(fl_server.history, key) for key in vars(fl_server.history))
    assert new_server.server_name == fl_server.server_name
    # After loading, new_server now has reporters, which were not present at initialization.
    assert isinstance(new_server.reports_manager.reporters[0], JsonReporter)  # Added for type checking
    assert isinstance(fl_server.reports_manager.reporters[0], JsonReporter)  # Added for type checking
    assert (
        new_server.reports_manager.reporters[0].metrics
        == fl_server.reports_manager.reporters[0].metrics
        == {
            "fit_start": "10.10",
            "host_type": "server",
        }
    )


def test_nnunet_server_state_checkpointer(tmp_path: Path) -> None:
    """Test the nnunet server per round state checkpointer."""

    def dummy_get_config(current_server_round: int) -> dict[str, Any]:
        return {
            "current_server_round": current_server_round,
        }

    nnunet_server = NnunetServer(
        client_manager=SimpleClientManager(),
        fl_config={"nnunet_config": "2d"},
        on_init_parameters_config_fn=dummy_get_config,
        server_name="nnunet_original_server",
    )
    nnunet_server_model = SingleLayerWithSeed(seed=42, output_size=1)
    dummy_plans = {"dataset_name": "x", "plans_name": "y"}
    nnunet_server.nnunet_plans_bytes = pickle.dumps(dummy_plans)
    nnunet_server.num_segmentation_heads = 2
    nnunet_server.num_input_channels = 3
    nnunet_server.current_round = 1
    nnunet_server.history = History()
    nnunet_server.global_deep_supervision = True
    # Temporary path to save the state to, will be cleaned up at the end of the test.
    checkpoint_dir = tmp_path.joinpath("nnunet_server_state")
    checkpoint_dir.mkdir()

    # Create a checkpointer for the server.
    checkpointer = NnUnetServerStateCheckpointer(
        checkpoint_dir=Path(checkpoint_dir), checkpoint_name="server_state.pt"
    )

    # Check that the checkpoint does not exist initially.
    assert not checkpointer.checkpoint_exists()

    # Save the state with the model.
    # In real usage, we should use parameter_exchanger to hydrate the model for checkpointing first.
    checkpointer.save_server_state(server=nnunet_server, model=nnunet_server_model)

    # Check that the checkpoint now exists.
    assert checkpointer.checkpoint_exists()

    # Create a new server to load the state into.
    new_server = NnunetServer(
        client_manager=SimpleClientManager(),
        fl_config={"nnunet_config": "3d_fullres"},
        on_init_parameters_config_fn=dummy_get_config,
        server_name="new_nnunet_server",
    )
    new_server_model = SingleLayerWithSeed(seed=12, output_size=1)
    dummy_plans = {"dataset_name": "x2", "plans_name": "y2"}
    new_server.nnunet_plans_bytes = pickle.dumps(dummy_plans)
    new_server.num_segmentation_heads = 1
    new_server.num_input_channels = 2
    new_server.history = History()
    new_server.current_round = 0

    # Check that new server is different from the original server.
    assert not all(
        torch.equal(value, nnunet_server_model.state_dict()[key])
        for key, value in new_server_model.state_dict().items()
    )
    # Check that new_server is different from nnunet_server.
    assert new_server.current_round != nnunet_server.current_round
    assert new_server.server_name != nnunet_server.server_name
    assert new_server.nnunet_plans_bytes != nnunet_server.nnunet_plans_bytes
    assert new_server.num_segmentation_heads != nnunet_server.num_segmentation_heads
    assert new_server.num_input_channels != nnunet_server.num_input_channels
    assert new_server.nnunet_config != nnunet_server.nnunet_config
    assert new_server.global_deep_supervision != nnunet_server.global_deep_supervision

    # Now load the state into the new server.
    # We don't need to create a new checkpointer here,
    # but we create one to simulate the situation where the program is
    # restarted due to an interruption. Note that checkpoint name and path should point to the saved checkpoint.
    new_checkpointer = NnUnetServerStateCheckpointer(
        checkpoint_dir=Path(checkpoint_dir), checkpoint_name="server_state.pt"
    )

    new_checkpointer.maybe_load_server_state(server=new_server, model=new_server_model)

    # Check that the state is loaded correctly.
    assert all(
        torch.equal(value, nnunet_server_model.state_dict()[key])
        for key, value in new_server_model.state_dict().items()
    )
    assert new_server.current_round == nnunet_server.current_round
    assert new_server.server_name == nnunet_server.server_name
    assert new_server.nnunet_plans_bytes == nnunet_server.nnunet_plans_bytes
    assert new_server.num_segmentation_heads == nnunet_server.num_segmentation_heads
    assert new_server.num_input_channels == nnunet_server.num_input_channels
    assert new_server.nnunet_config == nnunet_server.nnunet_config
    assert new_server.global_deep_supervision == nnunet_server.global_deep_supervision
