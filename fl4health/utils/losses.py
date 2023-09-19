from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List

import torch


@dataclass
class Losses:
    checkpoint: torch.Tensor
    backward: torch.Tensor
    additional_losses: Dict[str, torch.Tensor] = field(default_factory=lambda: {})


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

        num_losses = len(self.losses_list)
        # Compute average checkpoint and backward losses across list
        loss_dict["checkpoint"] = sum([losses.checkpoint.item() for losses in self.losses_list]) / num_losses
        loss_dict["backward"] = sum([losses.backward.item() for losses in self.losses_list]) / num_losses

        # We don't know the keys of the additional_losses beforehand so we extract them from the first entry
        # because we know all of the losses will have the same keys in additinal_losses dict
        for key in self.losses_list[0].additional_losses.keys():
            loss_dict[key] = sum([losses.additional_losses[key].item() for losses in self.losses_list]) / num_losses

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

        # Compute average checkpoint and backward losses across list
        loss_dict["checkpoint"] = sum([losses.checkpoint.item() for losses in self.losses_list])
        loss_dict["backward"] = sum([losses.backward.item() for losses in self.losses_list])

        # We don't know the keys of the additional_losses beforehand so we extract them from the first entry
        # because we know all of the losses will have the same keys in additinal_losses dict
        for key in self.losses_list[0].additional_losses.keys():
            loss_dict[key] = sum([losses.additional_losses[key].item() for losses in self.losses_list])

        return loss_dict
