from __future__ import annotations

import os
from abc import ABC, abstractmethod
from enum import Enum
from logging import ERROR, INFO, WARNING
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from flwr.common.logger import log
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


if TYPE_CHECKING:
    from fl4health.clients.basic_client import BasicClient
    from fl4health.servers.base_server import FlServer

from flwr.server.history import History

from fl4health.metrics.metric_managers import MetricManager
from fl4health.reporting.reports_manager import ReportsManager
from fl4health.utils.losses import LossMeter
from fl4health.utils.snapshotter import (
    AbstractSnapshotter,
    BytesSnapshotter,
    EnumSnapshotter,
    HistorySnapshotter,
    LRSchedulerSnapshotter,
    OptimizerSnapshotter,
    SerializableObjectSnapshotter,
    SingletonSnapshotter,
    StringSnapshotter,
    T,
    TorchModuleSnapshotter,
)


class StateCheckpointer(ABC):
    def __init__(
        self,
        checkpoint_dir: Path,
        checkpoint_name: str | None,
        snapshot_attrs: dict[str, tuple[AbstractSnapshotter, Any]],
    ) -> None:
        """
        Class for saving and loading the state of the client or server attributes. Attributes are stored in a
        dictionary to assist saving and are loaded in a dictionary. Checkpointing can be done after client or
        server round to facilitate restarting federated training if interrupted, or during the client's training
        loop to facilitate early stopping.

        Server and client state checkpointers will save to disk in the provided directory. A default name for the
        state checkpoint will be derived if checkpoint name remains none at the time of saving.

        Args:
            checkpoint_dir (Path): Directory to which checkpoints are saved. This can be modified later with
                ``set_checkpoint_path``
            checkpoint_name (str): Name of the checkpoint to be saved. If None at time of state saving, a default name
                will be given to the checkpoint. This can be changed later with ``set_checkpoint_path``
            snapshot_attrs (dict[str, tuple[AbstractSnapshotter, Any]]): Attributes that we need to save in order
                to allow for restarting of training.
        """
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_name = checkpoint_name
        self.checkpoint_path: str | None = None
        if self.checkpoint_name is not None:
            self.checkpoint_path = os.path.join(self.checkpoint_dir, self.checkpoint_name)

        self.snapshot_attrs = snapshot_attrs
        self.snapshot_ckpt: dict[str, Any] = {}

    def set_checkpoint_path(self, checkpoint_dir: Path, checkpoint_name: str) -> None:
        """
        Set or update the checkpoint path based on the provided checkpoint name and directory.

        Args:
            checkpoint_dir (Path): The directory where the checkpoint will be saved.
            checkpoint_name (str): The name of the checkpoint file.
        """
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_name = checkpoint_name
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
        assert self.checkpoint_path is not None, "Checkpoint path is not set but save_checkpoint has been called."
        try:
            log(INFO, f"Saving the state as {self.checkpoint_path}")
            torch.save(checkpoint_dict, self.checkpoint_path)
        except Exception as e:
            log(ERROR, f"Encountered the following error while saving the checkpoint: {e}")
            raise e

    def load_checkpoint(self) -> dict[str, Any]:
        """
        Load and return the checkpoint stored in ``checkpoint_dir`` under the  ``checkpoint_name`` if it exists. If
        it does not exist, an assertion error will be thrown.

        Returns:
            (dict[str, Any]): A dictionary representing the checkpointed state, as loaded by ``torch.load``.
        """
        assert self.checkpoint_path is not None, "Checkpoint path is not set but load_checkpoint has been called."
        assert self.checkpoint_exists(), f"Could not verify existence of checkpoint file at {self.checkpoint_path}"
        log(INFO, f"Loading state from checkpoint at {self.checkpoint_path}")

        return torch.load(self.checkpoint_path, weights_only=False)

    def checkpoint_exists(self) -> bool:
        """
        Check if a checkpoint exists at the ``checkpoint_path`` constructed as ``checkpoint_dir`` +
        ``checkpoint_name``.

        Returns:
            (bool): True if checkpoint exists, otherwise false.
        """
        assert self.checkpoint_path is not None, "A checkpoint_path should be set but is no"
        return os.path.exists(self.checkpoint_path)

    def add_to_snapshot_attr(self, name: str, snapshotter: AbstractSnapshotter, input_type: type[T]) -> None:
        """
        Add new attribute to the default ``snapshot_attrs`` dictionary. For this, we need a snapshotter that
        provides functionality for loading and saving the state of the attribute based on the type of the attribute.

        Args:
            name (str): Name of the attribute to be added.
            snapshotter (AbstractSnapshotter): Snapshotter object to be used for saving and loading the attribute.
            input_type (type[T]): Expected type of the attribute.
        """
        self.snapshot_attrs.update({name: (snapshotter, input_type)})

    def delete_from_snapshot_attr(self, name: str) -> None:
        """
        Delete the attribute from the default ``snapshot_attrs`` dictionary. This is useful for removing attributes
        that are no longer needed or to avoid saving/loading them.

        Args:
            name (str): Name of the attribute to be removed from the ``snapshot_attrs`` dictionary.
        """
        del self.snapshot_attrs[name]

    def save_state(self) -> None:
        """
        Create a snapshot of the state as defined in ``self.snapshot_attrs``.
        It is saved at ``self.checkpoint_path``.
        """
        for attr_name, (snapshotter, expected_type) in self.snapshot_attrs.items():
            self.snapshot_ckpt.update(self._save_snapshot(snapshotter, attr_name, expected_type))

        assert self.checkpoint_path is not None, "Attempting to save state but checkpoint_path is None"
        log(INFO, f"Saving the state to checkpoint at {self.checkpoint_path}")
        self.save_checkpoint(self.snapshot_ckpt)
        # Release snapshot memory after disk persistence
        self.snapshot_ckpt.clear()

    def load_state(self, attributes: list[str] | None = None) -> None:
        """
        Load checkpointed state dictionary from the checkpoint, potentially restricting the attributes to load.

        Args:
            attributes (list[str] | None): List of attributes to load from the checkpoint. If None, all attributes
                specified in ``snapshot_attrs`` are loaded. Defaults to None.
        """
        assert self.checkpoint_exists(), (
            f"No state checkpoint to load. Checkpoint at {self.checkpoint_path} does not exist"
        )

        if attributes is None:
            attributes = list(self.snapshot_attrs.keys())
            if not attributes:
                log(WARNING, "self.snapshot_attrs is empty, which may be undesired behavior.")

        # If the checkpoint exists, load it into snapshot_ckpt
        self.snapshot_ckpt = self.load_checkpoint()

        # Load components into target object
        for attr in attributes:
            snapshotter, expected_type = self.snapshot_attrs[attr]
            self._load_snapshot(snapshotter, attr, expected_type)
        log(INFO, f"Loaded the checkpointed state from {self.checkpoint_path}")

        # Release snapshot memory after loading
        self.snapshot_ckpt.clear()

    @abstractmethod
    def get_attribute(self, name: str) -> Any:
        """
        Get the attribute from the client or server.

        Args:
            name (str): Name of the attribute.

        Returns:
            (Any): The attribute value.
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
            (dict[str, T]): Wrapped attribute as a dictionary.
        """
        attribute = self.get_attribute(name)
        if isinstance(attribute, expected_type):
            return {"None": attribute}
        if isinstance(attribute, dict):
            for key, value in attribute.items():
                if not isinstance(value, expected_type):
                    raise ValueError(f"Incompatible type of attribute {type(attribute)} for key {key}")
            return attribute
        raise ValueError(f"Incompatible type of attribute {type(attribute)}, expected {expected_type}")

    def _save_snapshot(self, snapshotter: AbstractSnapshotter, name: str, expected_type: type[T]) -> dict[str, Any]:
        """
        Save the state of the attribute using the snapshotter's save_attribute functionality.

        Args:
            snapshotter (AbstractSnapshotter): Snapshotter object to save the attribute.
            name (str): Name of the attribute.
            expected_type (type[T]): Expected type of the attribute.

        Returns:
            (dict[str, Any]): A dictionary containing the state of the attribute.
        """
        attribute = self._dict_wrap_attr(name, expected_type)
        return {name: snapshotter.save_attribute(attribute)}

    def _load_snapshot(self, snapshotter: AbstractSnapshotter, name: str, expected_type: type[T]) -> None:
        """
        Load the state of the attribute using the snapshotter's ``load_attribute`` functionality.

        **NOTE**: This function assumes that ``snapshot_ckpt`` has been populated with the right data loaded from disk.

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


class ClientStateCheckpointer(StateCheckpointer):
    def __init__(
        self,
        checkpoint_dir: Path,
        checkpoint_name: str | None = None,
        snapshot_attrs: dict[str, tuple[AbstractSnapshotter, Any]] | None = None,
    ) -> None:
        """
        Class for saving and loading the state of a client's attributes as specified in ``snapshot_attrs``.

        Args:
            checkpoint_dir (Path): Directory to which checkpoints are saved. This can be modified later with
                ``set_checkpoint_path``
            checkpoint_name (str | None, optional): Name of the checkpoint to be saved. If None, but ``checkpoint_dir``
                is set then a default ``checkpoint_name`` based on the underlying name of the client to be
                checkpointed will be set of the form ``f"client_{client.client_name}_state.pt"``. This can be changed
                later with ``set_checkpoint_path``. Defaults to None.
            snapshot_attrs (dict[str, tuple[AbstractSnapshotter, Any]] | None, optional): Attributes that we need to
                save in order to allow for restarting of training. If None, a sensible default set of attributes and
                their associated snapshotters for an FL client are set. Defaults to None.
        """
        # If snapshot_attrs is None, we set a sensible default set of attributes to be saved. These are a minimal
        # set of attributes that can be used for per round checkpointing or early stopping.
        # NOTE: These default attributes are useful for state checkpointing a BasicClient. More sophisticated
        # clients may require more attributes to fully support training restarts and early stopping. For a server
        # example, see NnUnetServerStateCheckpointer.
        if snapshot_attrs is None:
            snapshot_attrs = {
                "model": (TorchModuleSnapshotter(), nn.Module),
                "optimizers": (OptimizerSnapshotter(), Optimizer),
                "lr_schedulers": (
                    LRSchedulerSnapshotter(),
                    LRScheduler,
                ),
                "total_steps": (SingletonSnapshotter(), int),
                "total_epochs": (SingletonSnapshotter(), int),
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

        super().__init__(checkpoint_dir, checkpoint_name, snapshot_attrs)
        self.client: BasicClient | None = None

    def maybe_set_default_checkpoint_name(self) -> None:
        """
        Potentially sets a default name for the checkpoint to be saved. If ``checkpoint_dir`` is set but
        ``checkpoint_name`` is None then a default ``checkpoint_name`` based on the underlying name of the client to
        be checkpointed will be set of the form ``f"client_{self.client.client_name}_state.pt"``.
        """
        assert self.client is not None, "Attempting to save client state but client is None"
        # Set the checkpoint name based on client's name if not already provided.
        if self.checkpoint_name is None:
            # If checkpoint_name is not provided, we set it based on the client name.
            self.checkpoint_name = f"client_{self.client.client_name}_state.pt"
            self.set_checkpoint_path(self.checkpoint_dir, self.checkpoint_name)

    def save_client_state(self, client: BasicClient) -> None:
        """
        Save the state of the client that is provided.

        Args:
            client (BasicClient): Client object with state to be saved.
        """
        # Store client for access in functions
        self.client = client
        # Potentially set a default checkpoint name
        self.maybe_set_default_checkpoint_name()
        # Saves everything in self.snapshot_attrs
        self.save_state()
        # Clear the client after being checkpointed.
        self.client = None

    def maybe_load_client_state(self, client: BasicClient, attributes: list[str] | None = None) -> bool:
        """
        Load the state into the client that is being provided.

        Args:
            client (BasicClient): Target client object into which state will be loaded
            attributes (list[str] | None, optional): List of attributes to load from the checkpoint. If None, all
                attributes specified in ``snapshot_attrs`` are loaded. Defaults to None.

        Returns:
            (bool): True if a checkpoint is successfully loaded. False otherwise.
        """
        # Store client for access in functions
        self.client = client
        # Setting default name if one doesn't exist. If we're here and it doesn't exist yet, user is expecting
        # a default name for loading anyway.
        self.maybe_set_default_checkpoint_name()
        if self.checkpoint_exists():
            self.load_state(attributes)
            log(INFO, f"State checkpoint successfully loaded from: {self.checkpoint_path}")
            # Clear the client after we are done updating its attributes.
            self.client = None
            return True

        log(INFO, f"No state checkpoint found at: {self.checkpoint_path}")
        # Clear the client since no checkpoint exists.
        self.client = None
        return False

    def get_attribute(self, name: str) -> Any:
        """
        Get the attribute from the client.

        Args:
            name (str): Name of the attribute.

        Returns:
            (Any): The attribute value.
        """
        assert self.client is not None, "Client is not set."
        return getattr(self.client, name)

    def set_attribute(self, name: str, value: Any) -> None:
        """
        Set the attribute on the client.

        Args:
            name (str): Name of the attribute.
            value (Any): Value to set for the attribute.
        """
        assert self.client is not None, "Client is not set."
        setattr(self.client, name, value)


class ServerStateCheckpointer(StateCheckpointer):
    def __init__(
        self,
        checkpoint_dir: Path,
        checkpoint_name: str | None = None,
        snapshot_attrs: dict[str, tuple[AbstractSnapshotter, Any]] | None = None,
    ) -> None:
        """
        Class for saving and loading the state of a server's attributes as specified in ``snapshot_attrs``.

        Args:
            checkpoint_dir (Path): Directory to which checkpoints are saved. This can be modified later with
                ``set_checkpoint_path``
            checkpoint_name (str | None, optional): Name of the checkpoint to be saved. If None, but ``checkpoint_dir``
                is set then a default ``checkpoint_name`` based on the underlying name of the client to be
                checkpointed will be set of the form ``f"server_{self.server.server_name}_state.pt"``. This can be
                updated later  with ``set_checkpoint_path``. Defaults to None.
            snapshot_attrs (dict[str, tuple[AbstractSnapshotter, Any]] | None, optional): Attributes that we need to
                save in order to allow for restarting of training. If None, a sensible default set of attributes and
                their associated snapshotters for an FL client are set. Defaults to None.
        """
        # If snapshot_attrs is None, we set a sensible default set of attributes to be saved. These are a minimal
        # set of attributes that can be used for per round checkpointing or early stopping.
        # NOTE: These default attributes are useful for state checkpointing a FlServer. More sophisticated servers
        # may require more attributes to fully support training restarts and early stopping. For an example, see
        # NnUnetServerStateCheckpointer.
        if snapshot_attrs is None:
            snapshot_attrs = {
                "model": (TorchModuleSnapshotter(), nn.Module),
                "current_round": (SingletonSnapshotter(), int),
                "reports_manager": (
                    SerializableObjectSnapshotter(),
                    ReportsManager,
                ),
                "server_name": (StringSnapshotter(), str),
                "history": (HistorySnapshotter(), History),
            }
        super().__init__(checkpoint_dir, checkpoint_name, snapshot_attrs)
        self.server: FlServer | None = None
        self.server_model: nn.Module | None = None

    def maybe_set_default_checkpoint_name(self) -> None:
        """
        Potentially sets a default name for the checkpoint to be saved. If ``checkpoint_dir`` is set but
        ``checkpoint_name`` is None then a default ``checkpoint_name`` based on the underlying name of the server to
        be checkpointed will be set of the form ``f"server_{self.server.server_name}_state.pt"``.
        """
        assert self.server is not None, "Attempting to save server state but server is None"
        # Set the checkpoint name based on server's name if not already provided.
        if self.checkpoint_name is None:
            # If checkpoint_name is not provided, we set it based on the server's name.
            self.checkpoint_name = f"server_{self.server.server_name}_state.pt"
            self.set_checkpoint_path(self.checkpoint_dir, self.checkpoint_name)

    def save_server_state(self, server: FlServer, model: nn.Module) -> None:
        """
        Save the state of the server, including a torch model, which is not a required component of the server class.

        Args:
            server (FlServer): Server with state to be saved
            model (nn.Module): The model to be saved as part of the server state.
        """
        # Store server and model for access in functions
        self.server = server
        # Server object does not have a model attribute, so we handle it separately.
        self.server_model = model
        # Potentially set a default checkpoint name
        self.maybe_set_default_checkpoint_name()
        # Saves everything in self.snapshot_attrs
        self.save_state()
        # Clear the server objects after checkpointing.
        self.server = None
        self.server_model = None

    def maybe_load_server_state(
        self, server: FlServer, model: nn.Module, attributes: list[str] | None = None
    ) -> nn.Module | None:
        """
        Load the state of the server from checkpoint.

        Args:
            server (FlServer): Server into which the attributes will be loaded.
            model (nn.Module): The model structure to be loaded as part of the server state.
            attributes (list[str] | None, optional): List of attributes to load from the checkpoint. If None, all
                attributes specified in ``snapshot_attrs`` are loaded. Defaults to None.

        Returns:
            (nn.Module | None): Returns a model if a checkpoint exists to load from. Otherwise returns None.
        """
        # Store server for access in functions
        self.server = server
        # Server object does not have a model attribute, so we handle it separately.
        self.server_model = model
        # Setting default name if one doesn't exist. If we're here and it doesn't exist yet, user is expecting
        self.maybe_set_default_checkpoint_name()
        if self.checkpoint_exists():
            self.load_state(attributes)
            log(INFO, f"State checkpoint successfully loaded from: {self.checkpoint_path}")
            # Clear the server after we are done updating its attributes.
            self.server = None
            # Server model is saved and returned separately for parameter extraction.
            return self.server_model

        log(INFO, f"No state checkpoint found at: {self.checkpoint_path}")
        # Clear the server object since checkpoint is not found.
        self.server = None
        self.server_model = None
        return None

    def get_attribute(self, name: str) -> Any:
        """
        Get the attribute from the server.

        Args:
            name (str): Name of the attribute.

        Returns:
            (Any): The attribute value.
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


class NnUnetServerStateCheckpointer(ServerStateCheckpointer):
    def __init__(
        self,
        checkpoint_dir: Path,
        checkpoint_name: str | None = None,
    ) -> None:
        """
        Class for saving and loading the state of the server's attributes based on the ``snapshot_attrs`` defined
        specifically for the nnUNet server.

        Args:
            checkpoint_dir (Path): Directory to which checkpoints are saved. This can be modified later with
                ``set_checkpoint_path``
            checkpoint_name (str | None, optional): Name of the checkpoint to be saved. If None, but ``checkpoint_dir``
                is set then a default ``checkpoint_name`` based on the underlying name of the client to be
                checkpointed will be set of the form ``f"server_{self.server.server_name}_state.pt"``. This can be
                updated later  with ``set_checkpoint_path``. Defaults to None.
        """
        # Go beyond default snapshot_attrs with nnUNet-specific attributes.
        nnunet_snapshot_attrs: dict[str, tuple[AbstractSnapshotter, Any]] = {
            "model": (TorchModuleSnapshotter(), nn.Module),
            "current_round": (SingletonSnapshotter(), int),
            "reports_manager": (
                SerializableObjectSnapshotter(),
                ReportsManager,
            ),
            "server_name": (StringSnapshotter(), str),
            "history": (HistorySnapshotter(), History),
            "nnunet_plans_bytes": (BytesSnapshotter(), bytes),
            "num_segmentation_heads": (SingletonSnapshotter(), int),
            "num_input_channels": (SingletonSnapshotter(), int),
            "global_deep_supervision": (EnumSnapshotter(), bool),
            "nnunet_config": (EnumSnapshotter(), Enum),
        }

        super().__init__(checkpoint_dir, checkpoint_name, snapshot_attrs=nnunet_snapshot_attrs)
