import copy
from abc import ABC, abstractmethod
from typing import Any, Generic, Type, TypeVar

import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from fl4health.clients.basic_client import BasicClient
from fl4health.reporting.reports_manager import ReportsManager
from fl4health.utils.losses import LossMeter
from fl4health.utils.metrics import MetricManager

T = TypeVar("T")


class Snapshotter(ABC, Generic[T]):
    def __init__(self, client: BasicClient) -> None:
        self.client = client

    def dict_wrap_attr(self, name: str, expected_type: Type[T]) -> dict[str, T]:
        attribute = copy.deepcopy(getattr(self.client, name))
        if isinstance(attribute, expected_type):
            return {"None": attribute}
        elif isinstance(attribute, dict):
            for key, value in attribute.items():
                if not isinstance(value, expected_type):
                    raise ValueError(f"Uncompatible type of attribute {type(attribute)} for key {key}")
            return attribute
        else:
            raise ValueError(f"Uncompatible type of attribute {type(attribute)}")

    def save(self, name: str, expected_type: Type[T]) -> Any:
        attribute = self.dict_wrap_attr(name, expected_type)
        return self.save_attribute(attribute)

    def load(self, ckpt: dict[str, Any], name: str, expected_type: Type[T]) -> None:
        attribute = self.dict_wrap_attr(name, expected_type)
        self.load_attribute(ckpt[name], attribute)
        if list(attribute.keys()) == ["None"]:
            setattr(self.client, name, attribute["None"])
        else:
            setattr(self.client, name, attribute)

    @abstractmethod
    def save_attribute(self, attribute: dict[str, T]) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def load_attribute(self, attribute_ckpt: dict[str, Any], attribute: dict[str, T]) -> None:
        raise NotImplementedError


class OptimizerSnapshotter(Snapshotter[Optimizer]):

    def save_attribute(self, attribute: dict[str, Optimizer]) -> dict[str, Any]:
        """
        Save the state of the optimizers (either single or dictionary of them).
        """
        output = {}
        for key, optimizer in attribute.items():
            output[key] = optimizer.state_dict()["state"]
        return output

    def load_attribute(self, attribute_ckpt: dict[str, Any], attribute: dict[str, Optimizer]) -> None:
        for key, optimizer in attribute.items():
            optimizer_state_dict = optimizer.state_dict()
            optimizer_state_dict["state"] = attribute_ckpt[key]
            optimizer.load_state_dict(optimizer_state_dict)


class LRSchedulerSnapshotter(Snapshotter[_LRScheduler]):

    def save_attribute(self, attribute: dict[str, _LRScheduler]) -> dict[str, Any]:
        """
        Save the state of the optimizers (either single or dictionary of them).
        """
        output = {}
        for key, lr_scheduler in attribute.items():
            output[key] = lr_scheduler.state_dict()
        return output

    def load_attribute(self, attribute_ckpt: dict[str, Any], attribute: dict[str, _LRScheduler]) -> None:
        for key, lr_scheduler in attribute.items():
            lr_scheduler.load_state_dict(attribute_ckpt[key])


class TorchModuleSnapshotter(Snapshotter[nn.Module]):

    def save_attribute(self, attribute: dict[str, nn.Module]) -> dict[str, Any]:
        """
        Save the state of the optimizers (either single or dictionary of them).
        """
        output = {}
        for key, model in attribute.items():
            output[key] = model.state_dict()
        return output

    def load_attribute(self, attribute_ckpt: dict[str, Any], attribute: dict[str, nn.Module]) -> None:
        for key, model in attribute.items():
            model.load_state_dict(attribute_ckpt[key])


class SerizableObjectSnapshotter(Snapshotter[MetricManager | LossMeter | ReportsManager]):
    def save_attribute(self, attribute: dict[str, MetricManager | LossMeter | ReportsManager]) -> dict[str, Any]:
        """
        Save the state of the optimizers (either single or dictionary of them).
        """
        return attribute

    def load_attribute(
        self, attribute_ckpt: dict[str, Any], attribute: dict[str, MetricManager | LossMeter | ReportsManager]
    ) -> None:
        for key in attribute:
            attribute[key] = attribute_ckpt[key]


class NumberSnapshotter(Snapshotter[int | float]):
    def save_attribute(self, attribute: dict[str, int | float]) -> dict[str, Any]:
        """
        Save the state of the optimizers (either single or dictionary of them).
        """
        return attribute

    def load_attribute(self, attribute_ckpt: dict[str, Any], attribute: dict[str, int | float]) -> None:
        for key in attribute:
            attribute[key] = attribute_ckpt[key]
