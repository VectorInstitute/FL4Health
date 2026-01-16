from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Generic, TypeVar

from flwr.server.history import History
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from fl4health.metrics.metric_managers import MetricManager
from fl4health.reporting.reports_manager import ReportsManager
from fl4health.utils.losses import LossMeter


T = TypeVar("T")


class AbstractSnapshotter(ABC, Generic[T]):
    @abstractmethod
    def save_attribute(self, attribute: dict[str, T]) -> dict[str, Any]:
        """
        Abstract method used to save the state of the attribute. This method should be implemented based on the type of
        the attribute and the way it should be saved.

        Args:
            attribute (dict[str, T]): The attribute to be saved.

        Returns:
            (dict[str, Any]): A dictionary containing the state of the attribute.
        """

    @abstractmethod
    def load_attribute(self, attribute_snapshot: dict[str, Any], attribute: dict[str, T]) -> None:
        """
        Abstract method to load the state of the attribute. This method should be implemented based on the type of
        the attribute and the way it should be loaded.

        Args:
            attribute_snapshot (dict[str, Any]): The snapshot containing the state of the attribute.
            attribute (dict[str, T]): The attribute to be loaded.
        """


class OptimizerSnapshotter(AbstractSnapshotter[Optimizer]):
    def save_attribute(self, attribute: dict[str, Optimizer]) -> dict[str, Any]:
        """
        Save the state of the optimizers by saving "state" attribute of the optimizers.

        Args:
            attribute (dict[str, Optimizer]): The optimizers to be saved.

        Returns:
            (dict[str, Any]): A dictionary containing the state of the optimizers.
        """
        output = {}
        for key, optimizer in attribute.items():
            output[key] = optimizer.state_dict()["state"]
        return output

    def load_attribute(self, attribute_snapshot: dict[str, Any], attribute: dict[str, Optimizer]) -> None:
        """
        Load the state of the optimizers by loading "state" attribute of the optimizers.

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
            (dict[str, Any]): A dictionary containing the state of the learning rate schedulers.
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
        Save the state of the ``nn.Modules``.

        Args:
            attribute (dict[str, nn.Module]): The ``nn.Modules`` to be saved.

        Returns:
            (dict[str, Any]): A dictionary containing the state of the ``nn.Modules``.
        """
        output = {}
        for key, model in attribute.items():
            output[key] = model.state_dict()
        return output

    def load_attribute(self, attribute_snapshot: dict[str, Any], attribute: dict[str, nn.Module]) -> None:
        """
        Load the state of the ``nn.Modules``.

        Args:
            attribute_snapshot (dict[str, Any]): The snapshot containing the state of the ``nn.Modules``.
            attribute (dict[str, nn.Module]): The ``nn.Modules`` to be loaded.
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
            (dict[str, Any]): A dictionary containing the state of the serializable objects.
        """
        return attribute

    def load_attribute(
        self, attribute_snapshot: dict[str, Any], attribute: dict[str, MetricManager | LossMeter | ReportsManager]
    ) -> None:
        """
        Load the state of the serializable objects (either single or dictionary of them).

        Args:
            attribute_snapshot (dict[str, Any]): The snapshot containing the state of the serializable objects.
            attribute (dict[str, MetricManager | LossMeter | ReportsManager]): The serializable objects to be loaded.
        """
        for key in attribute:
            attribute[key] = attribute_snapshot[key]


class SingletonSnapshotter(AbstractSnapshotter[int | float | bool]):
    def save_attribute(self, attribute: dict[str, int | float | bool]) -> dict[str, Any]:
        """
        Save the state of a singleton which could be a number or a boolean (either single or dictionary of them).

        Args:
            attribute (dict[str, int | float | bool]): The singleton to be saved.

        Returns:
            (dict[str, Any]): A dictionary containing the state of the singletons.
        """
        return attribute

    def load_attribute(self, attribute_snapshot: dict[str, Any], attribute: dict[str, int | float | bool]) -> None:
        """
        Load the state of the singleton (either single or dictionary of them).

        Args:
            attribute_snapshot (dict[str, Any]): The snapshot containing the state of the singleton.
            attribute (dict[str, int | float | bool]): The singletons to be loaded.
        """
        for key in attribute:
            attribute[key] = attribute_snapshot[key]


class HistorySnapshotter(AbstractSnapshotter[History]):
    def save_attribute(self, attribute: dict[str, History]) -> dict[str, Any]:
        """
        Save the state of the history objects (either single or dictionary of them).

        Args:
            attribute (dict[str, History]): The history to be saved.

        Returns:
            (dict[str, Any]): A dictionary containing the state of the history.
        """
        return attribute

    def load_attribute(self, attribute_snapshot: dict[str, Any], attribute: dict[str, History]) -> None:
        """
        Load the state of the history (either single or dictionary of them).

        Args:
            attribute_snapshot (dict[str, Any]): The snapshot containing the state of the history.
            attribute (dict[str, History]): The history to be loaded.
        """
        for key in attribute:
            attribute[key] = attribute_snapshot[key]


class StringSnapshotter(AbstractSnapshotter[str]):
    def save_attribute(self, attribute: dict[str, str]) -> dict[str, Any]:
        """
        Save the state of the strings (either single or dictionary of them).

        Args:
            attribute (dict[str, str]): The string to be saved.

        Returns:
            (dict[str, Any]): A dictionary containing the state of the strings.
        """
        return attribute

    def load_attribute(self, attribute_snapshot: dict[str, Any], attribute: dict[str, str]) -> None:
        """
        Load the state of the strings (either single or dictionary of them).

        Args:
            attribute_snapshot (dict[str, Any]): The snapshot containing the state of the strings.
            attribute (dict[str, str]): The strings to be loaded.
        """
        for key in attribute:
            attribute[key] = attribute_snapshot[key]


class BytesSnapshotter(AbstractSnapshotter[bytes]):
    def save_attribute(self, attribute: dict[str, bytes]) -> dict[str, Any]:
        """
        Save the state of the bytes (either single or dictionary of them).

        Args:
            attribute (dict[str, str]): The string to be saved.

        Returns:
            (dict[str, Any]): A dictionary containing the state of the bytes.
        """
        return attribute

    def load_attribute(self, attribute_snapshot: dict[str, Any], attribute: dict[str, bytes]) -> None:
        """
        Load the state of the bytes (either single or dictionary of them).

        Args:
            attribute_snapshot (dict[str, Any]): The snapshot containing the state of the bytes.
            attribute (dict[str, str]): The bytes to be loaded.
        """
        for key in attribute:
            attribute[key] = attribute_snapshot[key]


class EnumSnapshotter(AbstractSnapshotter[Enum]):
    def save_attribute(self, attribute: dict[str, Enum]) -> dict[str, Any]:
        """
        Save the state of the Enum (either single or dictionary of them).

        Args:
            attribute (dict[str, Enum]): The enum to be saved.

        Returns:
            (dict[str, Any]): A dictionary containing the state of the enum.
        """
        return attribute

    def load_attribute(self, attribute_snapshot: dict[str, Any], attribute: dict[str, Enum]) -> None:
        """
        Load the state of the num (either single or dictionary of them).

        Args:
            attribute_snapshot (dict[str, Any]): The snapshot containing the state of the enum.
            attribute (dict[str, Enum]): The enum to be loaded.
        """
        for key in attribute:
            attribute[key] = attribute_snapshot[key]
