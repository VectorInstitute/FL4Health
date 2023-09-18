from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from enum import Enum
from typing import Dict, List

import torch


@dataclass
class Losses:
    checkpoint: torch.Tensor
    backward: torch.Tensor


class LossMeterType(Enum):
    AVERAGE = "AVERAGE"
    ACCUMULATION = "ACCUMULATION"


class LossMeter(ABC):
    @abstractmethod
    def update(self, losses: Losses) -> None:
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def compute(self) -> Dict[str, float]:
        raise NotImplementedError

    @classmethod
    def get_meter_by_type(cls, meter_enum: LossMeterType) -> LossMeter:
        if meter_enum == LossMeterType.AVERAGE:
            return LossAverageMeter()
        elif meter_enum == LossMeterType.ACCUMULATION:
            return LossAccumulationMeter()
        else:
            raise ValueError(f"Not supported meter type: {str(meter_enum)}")


class LossAverageMeter(LossMeter):
    def __init__(self) -> None:
        self.losses_list: List[Losses] = []

    def update(self, losses: Losses) -> None:
        self.losses_list.append(losses)

    def clear(self) -> None:
        self.losses_list = []

    def compute(self) -> Dict[str, float]:
        assert len(self.losses_list) > 0
        loss_dict: Dict[str, float] = {}
        for field in fields(self.losses_list[0]):
            loss_dict[field.name] = sum([getattr(losses, field.name).item() for losses in self.losses_list]) / len(
                self.losses_list
            )

        return loss_dict


class LossAccumulationMeter(LossMeter):
    def __init__(self) -> None:
        self.losses_list: List[Losses] = []

    def update(self, losses: Losses) -> None:
        self.losses_list.append(losses)

    def clear(self) -> None:
        self.losses_list = []

    def compute(self) -> Dict[str, float]:
        assert len(self.losses_list) > 0
        loss_dict: Dict[str, float] = {}
        for field in fields(self.losses_list[0]):
            loss_dict[field.name] = sum([getattr(losses, field.name).item() for losses in self.losses_list])

        return loss_dict
