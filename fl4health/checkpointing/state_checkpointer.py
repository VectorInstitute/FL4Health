from __future__ import annotations

import copy
import os
from abc import ABC, abstractmethod
from enum import Enum
from logging import ERROR, INFO, WARNING
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from flwr.common.logger import log
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

if TYPE_CHECKING:
    from fl4health.clients.basic_client import BasicClient
    from fl4health.servers.base_server import FlServer

from flwr.server.history import History

from fl4health.reporting.reports_manager import ReportsManager
from fl4health.utils.losses import LossMeter
from fl4health.utils.metrics import MetricManager
from fl4health.utils.snapshotter import (
    AbstractSnapshotter,
    BytesSnapshotter,
    EnumSnapshotter,
    HistorySnapshotter,
    LRSchedulerSnapshotter,
    NumberSnapshotter,
    OptimizerSnapshotter,
    SerializableObjectSnapshotter,
    StringSnapshotter,
    T,
    TorchModuleSnapshotter,
)


class SimpleDictCheckpointer:
    """
    Simple state checkpointer for saving and loading the state of an object's attributes saved in a dictionary.
    This class is used to save and load the state of an object to a file. It is not meant to be used for
    saving in memory.
    """

    def __init__(
        self,
        checkpoint_dir: Path | None,
        checkpoint_name: str | None,
    ) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_name = checkpoint_name
        assert self.checkpoint_dir is not None, "Checkpoint directory should be set to facilitate saving on disk."
        self.save_on_disk = True
        self.checkpoint_path: str | None = None
        if self.checkpoint_name is not None:
            self.set_checkpoint_path(self.checkpoint_name, self.checkpoint_dir)

    def set_checkpoint_path(self, checkpoint_name: str, checkpoint_dir: Path | None = None) -> None:
        """
        Set or update the checkpoint path based on the provided checkpoint name and directory.

        Args:
            checkpoint_name (str): The name of the checkpoint file.
            checkpoint_dir (Path | None): The directory where the checkpoint will be saved.
        """
        self.checkpoint_name = checkpoint_name
        if checkpoint_dir is not None:
            # Also update the checkpoint dir.
            self.checkpoint_dir = checkpoint_dir
        assert self.checkpoint_dir is not None, "Checkpoint directory should be set to facilitate saving on disk."
        assert self.checkpoint_name is not None, "Checkpoint name should be set to facilitate saving on disk."
        self.checkpoint_path = os.path.join(self.checkpoint_dir, self.checkpoint_name)

    def save_checkpoint(self, checkpoint_dict: dict[str, Any]) -> None:
        """
        Save ``checkpoint_dict`` to checkpoint path defined based on checkpointer dir and checkpoint name.

        Args:
            checkpoint_dict (dict[str, Any]): A dictionary with string keys and values of type Any representing the
                state to be saved.

        Raises:
            e: Will throw an error if there is an issue saving the model. ``Torch.save`` seems to swallow errors in
                this context, so we explicitly surface the error with a try/except.
        """
        assert self.checkpoint_path is not None, "Checkpoint path is not set. Call set_checkpoint_path first."
        try:
            log(INFO, f"Saving the state as {self.checkpoint_path}")
            torch.save(checkpoint_dict, self.checkpoint_path)
        except Exception as e:
            log(ERROR, f"Encountered the following error while saving the checkpoint: {e}")
            raise e

    def load_checkpoint(self) -> dict[str, Any]:
        """
        Load and return the checkpoint stored in ``checkpoint_dir`` under
        the  ``checkpoint_name`` if it exists. If it does not exist, an assertion error will be thrown.

        Returns:
            dict[str, Any]: A dictionary representing the checkpointed state, as loaded by ``torch.load``.
        """
        assert self.checkpoint_path is not None, "Checkpoint path is not set. Call set_checkpoint_path first."
        assert self.checkpoint_exists()
        log(INFO, f"Loading state from checkpoint at {self.checkpoint_path}")

        return torch.load(self.checkpoint_path)

    def checkpoint_exists(self) -> bool:
        """
        Check if a checkpoint exists at the checkpoint_path constructed as
        ``checkpoint_dir`` + ``checkpoint_name`` during initialization.

        Returns:
            bool: True if checkpoint exists, otherwise false.
        """
        if self.save_on_disk:
            assert self.checkpoint_path is not None, "Checkpoint path is not set. Call set_checkpoint_path first."
            return os.path.exists(self.checkpoint_path)
        else:
            # Checkpoint is only saved in memory.
            return False


class StateCheckpointer(SimpleDictCheckpointer, ABC):

    def __init__(
        self,
        checkpoint_dir: Path | None,
        checkpoint_name: str | None,
        snapshot_attrs: dict[str, tuple[AbstractSnapshotter, Any]],
    ) -> None:
        """
        Class for saving and loading the state of the client or server attributes. Attributes are stored
        in a dictionary to assist saving and are loaded in a dictionary. Checkpointing can be done
        after client or server round to facilitate restarting federated training if interrupted,
        or during the client's training loop to facilitate early stopping.
        Server and client state checkpointers should be saved to the disk and not in memory to
        be recovered if training is interrupted. Client training loop state can optionally be saved in memory
        if no checkpointing_dir is provided (to facilitate early stopping).

        """
        self.checkpoint_dir: Path | None = checkpoint_dir
        self.checkpoint_name: str | None = checkpoint_name
        if checkpoint_dir is not None:
            # If checkpoint_dir is provided, state will be saved to the disk.
            self.checkpoint_dir = checkpoint_dir
            super().__init__(self.checkpoint_dir, self.checkpoint_name)
            self.save_on_disk = True
        else:
            # Otherwise, checkpoint_dir is None and the state will be saved in memory.
            log(
                WARNING,
                "State is being persisted in memory.\\"
                " You should provide a checkpoint_dir to facilitate restarting training if interrupted.",
            )
            self.save_on_disk = False

        self.snapshot_attrs = snapshot_attrs
        self.snapshot_ckpt: dict[str, Any] = {}

    @abstractmethod
    def get_attribute(self, name: str) -> Any:
        """
        Get the attribute from the client or server.

        Args:
            name (str): Name of the attribute.

        Returns:
            Any: The attribute value.
        """
        raise NotImplementedError("get_attribute must be implemented by inheriting classes")

    @abstractmethod
    def set_attribute(self, name: str, value: Any) -> None:
        """
        Set the attribute on the client or server.

        Args:
            name (str): Name of the attribute.
            value (Any): Value to set for the attribute.
        """
        raise NotImplementedError("set_attribute must be implemented by inheriting classes")

    def _dict_wrap_attr(self, name: str, expected_type: type[T]) -> dict[str, T]:
        """
        Wrap the attribute in a dictionary if it is not already a dictionary.

        Args:
            name (str): Name of the attribute.
            expected_type (type[T]): Expected type of the attribute.

        Returns:
            dict[str, T]: Wrapped attribute as a dictionary.
        """
        attribute = self.get_attribute(name)
        if isinstance(attribute, expected_type):
            return {"None": attribute}
        elif isinstance(attribute, dict):
            for key, value in attribute.items():
                if not isinstance(value, expected_type):
                    raise ValueError(f"Incompatible type of attribute {type(attribute)} for key {key}")
            return attribute
        else:
            raise ValueError(f"Incompatible type of attribute {type(attribute)}, expected {expected_type}")

    def _save_snapshot(self, snapshotter: AbstractSnapshotter, name: str, expected_type: type[T]) -> dict[str, Any]:
        """
        Save the state of the attribute using the snapshotter's save_attribute functionality.

        Args:
            snapshotter (AbstractSnapshotter): Snapshotter object to save the attribute.
            name (str): Name of the attribute.
            expected_type (type[T]): Expected type of the attribute.

        Returns:
            dict[str, Any]: A dictionary containing the state of the attribute.
        """
        attribute = self._dict_wrap_attr(name, expected_type)
        return {name: snapshotter.save_attribute(attribute)}

    def _load_snapshot(self, snapshotter: AbstractSnapshotter, name: str, expected_type: type[T]) -> None:
        """
        Load the state of the attribute to the client using the snapshotter's load_attribute functionality.

        Args:
            snapshotter (dict[str, Any]): Snapshotter object to return the state of the attribute.
            name (str): Name of the attribute.
            expected_type (type[T]): Expected type of the attribute.
        """
        attribute = self._dict_wrap_attr(name, expected_type)
        snapshotter.load_attribute(self.snapshot_ckpt[name], attribute)
        if list(attribute.keys()) == ["None"]:
            self.set_attribute(name, attribute["None"])
        else:
            self.set_attribute(name, attribute)

    def add_default_snapshot_attr(self, name: str, snapshotter: AbstractSnapshotter, input_type: type[T]) -> None:
        """
        Add new attribute to the default snapshot_attrs dictionary. For this, we need a snapshotter that
        provides functionality for loading and saving the state of the attribute based on the type of the attribute.

        Args:
            name (str): Name of the attribute to be added.
            snapshotter (AbstractSnapshotter): Snapshotter object to be used for saving and loading the attribute.
            input_type (type[T]): Expected type of the attribute.
        """
        self.snapshot_attrs.update({name: (snapshotter, input_type)})

    def delete_default_snapshot_attr(self, name: str) -> None:
        """
        Delete the attribute from the default snapshot_attrs dictionary.
        This is useful for removing attributes that are no longer needed or to avoid saving/loading them.

        Args:
            name (str): Name of the attribute to be removed from the ``snapshot_attrs`` dictionary.
        """
        del self.snapshot_attrs[name]

    def save_state(self) -> None:
        """
        Create a snapshot of the state as defined in self.snapshot_attrs.
        It is either saved in the ``checkpoint_dir`` (if provided) under ``checkpoint_name``,
        or remains in memory (self.snapshot_ckpt).
        """
        for attr_name, (snapshotter, expected_type) in self.snapshot_attrs.items():
            self.snapshot_ckpt.update(self._save_snapshot(snapshotter, attr_name, expected_type))

        if self.checkpoint_dir is not None:
            log(
                INFO,
                f"Saving the state to checkpoint at {self.checkpoint_dir} " f"with name {self.checkpoint_name}.",
            )
            self.save_checkpoint(self.snapshot_ckpt)
            self.snapshot_ckpt.clear()

        else:
            log(
                WARNING,
                "Checkpointing directory is not provided. State will be kept in the memory.",
            )
            self.snapshot_ckpt = copy.deepcopy(self.snapshot_ckpt)

    def load_state(self, attributes: list[str] | None = None) -> None:
        """
        Load checkpointed state dictionary either from the checkpoint or from the memory (self.snapshot_attrs).

        Args:
            attributes (list[str] | None): List of attributes to load from the checkpoint. If None, all attributes
                specified in ``snapshot_attrs`` are loaded. Defaults to None.
        """
        assert self.checkpoint_exists() or self.snapshot_ckpt != {}, "No state checkpoint to load."

        if attributes is None:
            attributes = list(self.snapshot_attrs.keys())

        # If the checkpoint exists, load it, otherwise load from self.snapshot_ckpt (in memory).
        if self.checkpoint_dir is not None and self.checkpoint_exists():
            self.snapshot_ckpt = self.load_checkpoint()

        for attr in attributes:
            snapshotter, expected_type = self.snapshot_attrs[attr]
            self._load_snapshot(snapshotter, attr, expected_type)


class ClientStateCheckpointer(StateCheckpointer):

    def __init__(
        self,
        checkpoint_dir: Path | None,
        checkpoint_name: str | None,
        snapshot_attrs: dict[str, tuple[AbstractSnapshotter, Any]],
    ) -> None:
        """
        Class for saving and loading the state of the client's attributes as specified in ``snapshot_attrs``.
        """
        self.snapshot_attrs = snapshot_attrs
        super().__init__(
            checkpoint_dir,
            checkpoint_name,
            self.snapshot_attrs,
        )
        self.client: BasicClient | None = None

    def set_client(self, client: BasicClient) -> None:
        """
        Set the client to be monitored.

        Args:
            client (BasicClient): The client to be monitored.
        """
        # First hold the client in memory for checkpointing and loading.
        self.client = client
        self.client_name = client.client_name
        # Set the checkpoint name based on client's name if not already provided.
        if self.checkpoint_name is None and self.checkpoint_dir is not None:
            # If checkpoint_name is not provided, we set it based on the client name.
            self.checkpoint_name = f"client_{self.client_name}_state.pt"
            self.set_checkpoint_path(self.checkpoint_name)

    def save_client_state(self) -> None:
        """
        Save the state of the client that is being monitored. Client is set in ``set_client``.
        """
        assert self.client is not None, "Client is not set. First call set_client."
        # Saves everything in self.snapshot_attrs
        self.save_state()
        # Clear the client after being checkpointed.
        self.client = None

    def load_client_state(self, attributes: list[str] | None = None) -> None:
        """
        Load the state of the client that is being monitored. Client is set in ``set_client``.

        Args:
            attributes (list[str] | None): List of attributes to load from the checkpoint. If None, all attributes
                specified in ``snapshot_attrs`` are loaded. Defaults to None.
        """
        assert self.client is not None, "Client is not set. First call set_client."
        self.load_state(attributes)
        # Clear the client after we are done updating its attributes.
        self.client = None

    def get_attribute(self, name: str) -> Any:
        """
        Get the attribute from the client.

        Args:
            name (str): Name of the attribute.

        Returns:
            Any: The attribute value.
        """
        assert self.client is not None, "Client is not set."
        attribute = getattr(self.client, name)
        return attribute

    def set_attribute(self, name: str, value: Any) -> None:
        """
        Set the attribute on the client.

        Args:
            name (str): Name of the attribute.
            value (Any): Value to set for the attribute.
        """
        assert self.client is not None, "Client is not set."
        setattr(self.client, name, value)


class ClientPerRoundStateCheckpointer(ClientStateCheckpointer):
    def __init__(
        self,
        checkpoint_dir: Path,
        checkpoint_name: str | None = None,
    ) -> None:
        """
        Abstract class for saving and loading the state of the client's attributes
        after each FL round.
        """
        assert (
            checkpoint_dir is not None
        ), "Checkpoint directory should be set for\
            per round checkpointing to facilitate restarting federated training if interrupted."
        # We don't need to save the client model here since parameters are loaded from the server at the start
        # of the round.
        self.snapshot_attrs: dict = {
            "optimizers": (OptimizerSnapshotter(), Optimizer),
            "lr_schedulers": (
                LRSchedulerSnapshotter(),
                LRScheduler,
            ),
            "total_steps": (NumberSnapshotter(), int),
            "total_epochs": (NumberSnapshotter(), int),
            "reports_manager": (
                SerializableObjectSnapshotter(),
                ReportsManager,
            ),
        }
        # No default checkpoint_name is provided.
        # It should be user-defined and specific for each clients, or it will be set when client is set.
        super().__init__(
            checkpoint_dir,
            checkpoint_name,
            self.snapshot_attrs,
        )


class ClientTrainLoopCheckpointer(ClientStateCheckpointer):

    def __init__(
        self,
        checkpoint_dir: Path | None = None,
        checkpoint_name: str | None = None,
    ) -> None:
        """
        Class for saving and loading the state of the client's attributes
        in the training loop. It is possible to save the state in memory
        if no checkpoint_dir is provided, or to save it to the disk if
        checkpoint_dir is provided.

        """
        # snapshot_attrs specifies the attributes that will be saved and loaded
        # to facilitate checkpointing inside the training loop (used for early stopping).
        self.snapshot_attrs: dict = {
            "model": (TorchModuleSnapshotter(), nn.Module),
            "optimizers": (OptimizerSnapshotter(), Optimizer),
            "lr_schedulers": (
                LRSchedulerSnapshotter(),
                LRScheduler,
            ),
            "total_steps": (NumberSnapshotter(), int),
            "total_epochs": (NumberSnapshotter(), int),
            "reports_manager": (
                SerializableObjectSnapshotter(),
                ReportsManager,
            ),
            "train_loss_meter": (
                SerializableObjectSnapshotter(),
                LossMeter,
            ),
            "train_metric_manager": (
                SerializableObjectSnapshotter(),
                MetricManager,
            ),
        }
        # No default checkpoint_name is provided.
        # It should be user-defined and specific for each clients, or it will be set when client is set.
        super().__init__(
            checkpoint_dir,
            checkpoint_name,
            self.snapshot_attrs,
        )


class ServerStateCheckpointer(StateCheckpointer):

    def __init__(
        self,
        checkpoint_dir: Path | None,
        checkpoint_name: str | None,
        snapshot_attrs: dict[str, tuple[AbstractSnapshotter, Any]],
    ) -> None:
        """
        Class for saving and loading the state of the server's attributes as specified in ``snapshot_attrs``.
        """
        self.snapshot_attrs = snapshot_attrs
        super().__init__(
            checkpoint_dir,
            checkpoint_name,
            self.snapshot_attrs,
        )
        self.server: FlServer | None = None
        self.server_model: nn.Module | None = None

    def set_server(self, server: FlServer) -> None:
        """
        Set the server to be monitored.

        Args:
            server (FlServer): The server to be monitored.
        """
        # First hold the server in memory for checkpointing and loading.
        self.server = server
        self.server_name = server.server_name
        # Set the checkpoint name based on server's name if not already provided.
        if self.checkpoint_name is None:
            self.state_checkpoint_name = f"server_{self.server_name}_state.pt"

    def save_server_state(self, model: nn.Module) -> None:
        """
        Save the state of the server.

        Args:
            model (nn.Module): The model to be saved as part of the server state.
        """
        assert self.server is not None, "Server is not set. First call set_server."
        # Server object does not have a model attribute, so we handle it separately.
        assert model is not None, "Model should be provided to save the server state."
        self.server_model = model
        self.save_state()
        # Clear the server object after checkpointing.
        self.server = None

    def load_server_state(self, model: nn.Module, attributes: list[str] | None = None) -> nn.Module:
        """
        Load the state of the server from checkpoint.

        Args:
            model nn.Module: The model structure to be loaded as part of the server state.
            attributes (list[str] | None): List of attributes to load from the checkpoint. If None, all attributes
                specified in ``snapshot_attrs`` are loaded. Defaults to None.
        """
        assert self.server is not None, "Server is not set. First call set_server."
        self.server_model = model
        self.load_state(attributes)
        # Server object does not have a model attribute, so we handle it separately.
        # Server model is saved and returned separately for parameter extraction.
        # Clear the server after we are done updating its attributes.
        self.server = None
        # Return the server model.
        return self.server_model

    def get_attribute(self, name: str) -> Any:
        """
        Get the attribute from the server.

        Args:
            name (str): Name of the attribute.

        Returns:
            Any: The attribute value.
        """
        assert self.server is not None, "Server is not set."
        if name == "model":
            return self.server_model
        return getattr(self.server, name)

    def set_attribute(self, name: str, value: Any) -> None:
        """
        Set the attribute on the server.

        Args:
            name (str): Name of the attribute.
            value (Any): Value to set for the attribute.
        """
        assert self.server is not None, "Server is not set."
        if name == "model":
            self.server_model = value
        else:
            setattr(self.server, name, value)


class ServerPerRoundStateCheckpointer(ServerStateCheckpointer):

    def __init__(
        self,
        checkpoint_dir: Path,
        checkpoint_name: str | None = None,
    ) -> None:
        """
        Class for saving and loading the state of the server's attributes after each FL round.
        This class is used to save and load the state of the server to the disk to facilitate
        restarting federated training if interrupted.
        """
        assert (
            checkpoint_dir is not None
        ), "Checkpoint directory should be set for\
            per round checkpointing to facilitate restarting federated training if interrupted."
        if checkpoint_name is None:
            checkpoint_name = "server_state_checkpoint.pt"

        self.snapshot_attrs: dict = {
            "model": (TorchModuleSnapshotter(), nn.Module),
            "current_round": (NumberSnapshotter(), int),
            "reports_manager": (
                SerializableObjectSnapshotter(),
                ReportsManager,
            ),
            "server_name": (StringSnapshotter(), str),
            "history": (HistorySnapshotter(), History),
        }
        super().__init__(
            checkpoint_dir,
            checkpoint_name,
            self.snapshot_attrs,
        )


class NnUnetServerPerRoundStateCheckpointer(ServerPerRoundStateCheckpointer):

    def __init__(
        self,
        checkpoint_dir: Path,
        checkpoint_name: str | None = None,
    ) -> None:
        """
        Class for saving and loading the state of the server's attributes based on
        the ``snapshot_attrs`` defined specifically for the nnUNet server.

        """
        assert (
            checkpoint_dir is not None
        ), "Checkpoint directory should be set for\
            per round checkpointing to facilitate restarting federated training if interrupted."
        if checkpoint_name is None:
            checkpoint_name = "nnunet_server_state_checkpoint.pt"

        super().__init__(
            checkpoint_dir,
            checkpoint_name,
        )
        # Override's parent class's snapshot_attrs with nnUNet-specific attributes.
        self.snapshot_attrs: dict = {
            "model": (TorchModuleSnapshotter(), nn.Module),
            "current_round": (NumberSnapshotter(), int),
            "reports_manager": (
                SerializableObjectSnapshotter(),
                ReportsManager,
            ),
            "server_name": (StringSnapshotter(), str),
            "history": (HistorySnapshotter(), History),
            "nnunet_plans_bytes": (BytesSnapshotter(), bytes),
            "num_segmentation_heads": (NumberSnapshotter(), int),
            "num_input_channels": (NumberSnapshotter(), int),
            "global_deep_supervision": (EnumSnapshotter(), bool),
            "nnunet_config": (EnumSnapshotter(), Enum),
        }
