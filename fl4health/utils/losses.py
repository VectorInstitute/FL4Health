from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Optional

import torch


class Losses:
    def __init__(
        self,
        checkpoint: torch.Tensor,
        backward: torch.Tensor,
        additional_losses: Optional[Dict[str, torch.Tensor]] = None,
    ):
        self.checkpoint = checkpoint
        self.backward = backward
        self.additional_losses = additional_losses

    def as_dict(self) -> Dict[str, float]:
        loss_dict: Dict[str, float] = {}
        loss_dict["checkpoint"] = float(self.checkpoint.item())
        loss_dict["backward"] = float(self.backward.item())

        if self.additional_losses is not None:
            for key, val in self.additional_losses.items():
                loss_dict[key] = float(val.item())

        return loss_dict


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
    def compute(self) -> Losses:
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

    def compute(self) -> Losses:
        assert len(self.losses_list) > 0

        num_losses = len(self.losses_list)
        # Compute average checkpoint and backward losses across list
        checkpoint_loss = torch.sum(torch.FloatTensor([losses.checkpoint for losses in self.losses_list])) / num_losses
        backward_loss = torch.sum(torch.FloatTensor([losses.backward for losses in self.losses_list])) / num_losses

        # We don't know the keys of the additional_losses beforehand so we first check if it is none first
        # If it is not none, we iterate through the keys, compute the additional losses and store
        # else additional_losses is set to None
        if self.losses_list[0].additional_losses is not None:
            additional_losses = {}
            for key in self.losses_list[0].additional_losses.keys():
                additional_losses[key] = (
                    torch.sum(
                        torch.FloatTensor(
                            [
                                losses.additional_losses[key]
                                for losses in self.losses_list
                                if losses.additional_losses is not None and key in losses.additional_losses
                            ]
                        )
                    )
                    / num_losses
                )
        else:
            additional_losses = None

        losses = Losses(checkpoint=checkpoint_loss, backward=backward_loss, additional_losses=additional_losses)
        return losses


class LossAccumulationMeter(LossMeter):
    def __init__(self) -> None:
        self.losses_list: List[Losses] = []

    def update(self, losses: Losses) -> None:
        self.losses_list.append(losses)

    def clear(self) -> None:
        self.losses_list = []

    def compute(self) -> Losses:
        assert len(self.losses_list) > 0

        # Compute average checkpoint and backward losses across list
        checkpoint_loss = torch.sum(torch.FloatTensor([losses.checkpoint for losses in self.losses_list]))
        backward_loss = torch.sum(torch.FloatTensor([losses.backward for losses in self.losses_list]))

        # We don't know the keys of the additional_losses beforehand so we first check if it is none first
        # If it is not none, we iterate through the keys, compute the additional losses and store
        # else additional_losses is set to None
        if self.losses_list[0].additional_losses is not None:
            additional_losses = {}
            for key in self.losses_list[0].additional_losses.keys():
                additional_losses[key] = torch.sum(
                    torch.FloatTensor(
                        [
                            losses.additional_losses[key]
                            for losses in self.losses_list
                            if losses.additional_losses is not None and key in losses.additional_losses
                        ]
                    )
                )
        else:
            additional_losses = None

        losses = Losses(checkpoint=checkpoint_loss, backward=backward_loss, additional_losses=additional_losses)
        return losses
