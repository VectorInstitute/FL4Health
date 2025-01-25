from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Generic, TypeVar

import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

if TYPE_CHECKING:
    from fl4health.clients.basic_client import BasicClient

from fl4health.reporting.reports_manager import ReportsManager
from fl4health.utils.losses import LossMeter
from fl4health.utils.metrics import MetricManager

T = TypeVar("T")


class AbstractSnapshotter(ABC, Generic[T]):
    def __init__(self, client: BasicClient) -> None:
        """
        Abstract class for saving and loading the state of the client's attributes.

        Args:
            client (BasicClient): The client to be monitored.
        """
        self.client = client

    def dict_wrap_attr(self, name: str, expected_type: type[T]) -> dict[str, T]:
        """
        Wrap the attribute in a dictionary if it is not already a dictionary.

        Args:
            name (str): Name of the attribute.
            expected_type (type[T]): Expected type of the attribute.

        Returns:
            dict[str, T]: Wrapped attribute as a dictionary.
        """
        attribute = getattr(self.client, name)
        if isinstance(attribute, expected_type):
            return {"None": attribute}
        elif isinstance(attribute, dict):
            for key, value in attribute.items():
                if not isinstance(value, expected_type):
                    raise ValueError(f"Incompatible type of attribute {type(attribute)} for key {key}")
            return attribute
        else:
            raise ValueError(f"Incompatible type of attribute {type(attribute)}")

    def save(self, name: str, expected_type: type[T]) -> dict[str, Any]:
        """
        Save the state of the attribute.

        Args:
            name (str): Name of the attribute.
            expected_type (type[T]): Expected type of the attribute.

        Returns:
            dict[str, Any]: A dictionary containing the state of the attribute.
        """
        attribute = self.dict_wrap_attr(name, expected_type)
        return {name: self.save_attribute(attribute)}

    def load(self, snapshot: dict[str, Any], name: str, expected_type: type[T]) -> None:
        """
        Load the state of the attribute to the client.

        Args:
            snapshot (dict[str, Any]): Snapshot containing the state of the attribute.
            name (str): Name of the attribute.
            expected_type (type[T]): Expected type of the attribute.
        """
        attribute = self.dict_wrap_attr(name, expected_type)
        self.load_attribute(snapshot[name], attribute)
        if list(attribute.keys()) == ["None"]:
            setattr(self.client, name, attribute["None"])
        else:
            setattr(self.client, name, attribute)

    @abstractmethod
    def save_attribute(self, attribute: dict[str, T]) -> dict[str, Any]:
        """
        Abstract method to save the state of the attribute. This method should be implemented
        based on the type of the attribute and the way it should be saved.

        Args:
            attribute (dict[str, T]): The attribute to be saved.

        Returns:
            dict[str, Any]: A dictionary containing the state of the attribute.
        """

    @abstractmethod
    def load_attribute(self, attribute_snapshot: dict[str, Any], attribute: dict[str, T]) -> None:
        """
        Abstract method to load the state of the attribute. This method should be implemented
        based on the type of the attribute and the way it should be loaded.

        Args:
            attribute_snapshot (dict[str, Any]): The snapshot containing the state of the attribute.
            attribute (dict[str, T]): The attribute to be loaded.
        """


class OptimizerSnapshotter(AbstractSnapshotter[Optimizer]):

    def save_attribute(self, attribute: dict[str, Optimizer]) -> dict[str, Any]:
        """
        Save the state of the optimizers by saving "state" attribute of the optimizer.

        Args:
            attribute (dict[str, Optimizer]): The optimizers to be saved.

        Returns:
            dict[str, Any]: A dictionary containing the state of the optimizers.
        """
        output = {}
        for key, optimizer in attribute.items():
            output[key] = optimizer.state_dict()["state"]
        return output

    def load_attribute(self, attribute_snapshot: dict[str, Any], attribute: dict[str, Optimizer]) -> None:
        """
        Load the state of the optimizers by loading "state" attribute of the optimizer

        Args:
            attribute_snapshot (dict[str, Any]): The snapshot containing the state of the optimizers.
            attribute (dict[str, Optimizer]): The optimizers to be loaded.
        """
        for key, optimizer in attribute.items():
            optimizer_state_dict = optimizer.state_dict()
            optimizer_state_dict["state"] = attribute_snapshot[key]
            optimizer.load_state_dict(optimizer_state_dict)


class LRSchedulerSnapshotter(AbstractSnapshotter[LRScheduler]):

    def save_attribute(self, attribute: dict[str, LRScheduler]) -> dict[str, Any]:
        """
        Save the state of the learning rate schedulers.

        Args:
            attribute (dict[str, LRScheduler]): The learning rate schedulers to be saved.

        Returns:
            dict[str, Any]: A dictionary containing the state of the learning rate schedulers.
        """
        output = {}
        for key, lr_scheduler in attribute.items():
            output[key] = lr_scheduler.state_dict()
        return output

    def load_attribute(self, attribute_snapshot: dict[str, Any], attribute: dict[str, LRScheduler]) -> None:
        """
        Load the state of the learning rate schedulers.

        Args:
            attribute_snapshot (dict[str, Any]): The snapshot containing the state of the learning rate schedulers.
            attribute (dict[str, LRScheduler]): The learning rate schedulers to be loaded.
        """
        for key, lr_scheduler in attribute.items():
            lr_scheduler.load_state_dict(attribute_snapshot[key])


class TorchModuleSnapshotter(AbstractSnapshotter[nn.Module]):

    def save_attribute(self, attribute: dict[str, nn.Module]) -> dict[str, Any]:
        """
        Save the state of the nn.Modules.

        Args:
            attribute (dict[str, nn.Module]): The nn.Modules to be saved.

        Returns:
            dict[str, Any]: A dictionary containing the state of the nn.Modules.
        """
        output = {}
        for key, model in attribute.items():
            output[key] = model.state_dict()
        return output

    def load_attribute(self, attribute_snapshot: dict[str, Any], attribute: dict[str, nn.Module]) -> None:
        """
        Load the state of the nn.Modules.

        Args:
            attribute_snapshot (dict[str, Any]): The snapshot containing the state of the nn.Modules.
            attribute (dict[str, nn.Module]): The nn.Modules to be loaded
        """
        for key, model in attribute.items():
            model.load_state_dict(attribute_snapshot[key])


class SerializableObjectSnapshotter(AbstractSnapshotter[MetricManager | LossMeter | ReportsManager]):
    def save_attribute(self, attribute: dict[str, MetricManager | LossMeter | ReportsManager]) -> dict[str, Any]:
        """
        Save the state of the serializable objects (either single or dictionary of them).

        Args:
            attribute (dict[str, MetricManager | LossMeter | ReportsManager]): The serializable objects to be saved.

        Returns:
            dict[str, Any]: A dictionary containing the state of the serializable objects.
        """
        return attribute

    def load_attribute(
        self, attribute_snapshot: dict[str, Any], attribute: dict[str, MetricManager | LossMeter | ReportsManager]
    ) -> None:
        """
        Load the state of the serializable objects (either single or dictionary of them).

        Args:
            attribute_snapshot (dict[str, Any]): The snapshot containing the state of the serializable objects.
            attribute (dict[str, MetricManager | LossMeter | ReportsManager]): The serializable objects to be loaded
        """
        for key in attribute:
            attribute[key] = attribute_snapshot[key]


class NumberSnapshotter(AbstractSnapshotter[int | float]):
    def save_attribute(self, attribute: dict[str, int | float]) -> dict[str, Any]:
        """
        Save the state of the numbers (either single or dictionary of them).

        Args:
            attribute (dict[str, int | float]): The numbers to be saved.

        Returns:
            dict[str, Any]: A dictionary containing the state of the numbers.
        """
        return attribute

    def load_attribute(self, attribute_snapshot: dict[str, Any], attribute: dict[str, int | float]) -> None:
        """
        Load the state of the numbers (either single or dictionary of them).

        Args:
            attribute_snapshot (dict[str, Any]): The snapshot containing the state of the numbers.
            attribute (dict[str, int | float]): The numbers to be loaded
        """
        for key in attribute:
            attribute[key] = attribute_snapshot[key]
