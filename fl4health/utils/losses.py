from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Optional, Sequence, Union

import torch


class Losses:
    def __init__(
        self,
        checkpoint: torch.Tensor,
        backward: Union[torch.Tensor, Dict[str, torch.Tensor]],
        additional_losses: Optional[Dict[str, torch.Tensor]] = None,
    ) -> None:
        """
        A class to store the checkpoint, backward and additional_losses of a model
        along with a method to return a dictionary representation.

        Args:
            checkpoint (torch.Tensor): The loss used to checkpoint model (if checkpointing is enabled).
            backward (Union[torch.Tensor, Dict[str, torch.Tensor]]): The backward loss or
                losses to optimize. In the normal case, backward is a Tensor corresponding to the
                loss of a model. In the case of an ensemble_model, backward is dictionary of losses.
            additional_losses (Optional[Dict[str, torch.Tensor]]): Optional dictionary of additional losses.
        """
        self.checkpoint = checkpoint
        self.backward = backward if isinstance(backward, dict) else {"backward": backward}
        self.additional_losses = additional_losses if additional_losses else {}

    def as_dict(self) -> Dict[str, float]:
        """
        Produces a dictionary representation of the object with all of the losses.

        Returns:
            Dict[str, float]: A dictionary where each key represents one of the checkpoint,
                backward or additional losses.
        """
        loss_dict: Dict[str, float] = {}
        loss_dict["checkpoint"] = float(self.checkpoint.item())

        backward = {key: float(loss.item()) for key, loss in self.backward.items()}
        loss_dict.update(backward)

        if self.additional_losses is not None:
            for key, val in self.additional_losses.items():
                loss_dict[key] = float(val.item())

        return loss_dict


class LossMeterType(Enum):
    AVERAGE = "AVERAGE"
    ACCUMULATION = "ACCUMULATION"


class LossType(Enum):
    BACKWARD = "BACKWARD"
    ADDITIONAL = "ADDITIONAL"


class LossMeter(ABC):
    @abstractmethod
    def update(self, losses: Losses) -> None:
        """
        Appends loss to list of losses.

        Args:
            losses (Losses): A losses object with checkpoint, backward and additional losses.

        Raises:
            NotImplementedError: To be implented by child classes.
        """
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        """
        Reset the meter by re-initializing losses_list to be empty

        Raises:
            NotImplementedError: To be implemented by child class.
        """
        raise NotImplementedError

    @abstractmethod
    def compute(self) -> Losses:
        """
        Compute aggregation of current list of losses if non-empty.

        Returns:
            Losses: New Losses object with aggregate of losses in losses_list.

        Raises:
            NotImplentedError: To be impemented by child class.
        """
        raise NotImplementedError

    @classmethod
    def get_meter_by_type(cls, meter_enum: LossMeterType) -> LossMeter:
        """
        Class method that returns LossMeter instance of given type.

        Args:
            meter_enum (LossMeterType): The type of loss meter to create.

        Returns:
            LossMeter: New LossMeter instance of given type.
        """
        if meter_enum == LossMeterType.AVERAGE:
            return LossAverageMeter()
        elif meter_enum == LossMeterType.ACCUMULATION:
            return LossAccumulationMeter()
        else:
            raise ValueError(f"Not supported meter type: {str(meter_enum)}")

    @classmethod
    def aggregate_loss_by_type(
        cls, losses_list: Sequence[Losses], loss_type: LossType, loss_meter_type: LossMeterType
    ) -> Dict[str, torch.Tensor]:
        """
        Class method that aggregates a loss by loss_type (ie backward, checkpoint, additional).

        Args:
            losses_list (Sequence[Losses]): List of losses to aggregate.
            loss_type (LossType): The type of loss (ie backward or additional loss).
            loss_meter_type (LossMeterType): The type of LossMeter (ie average or accumulation).

        Returns:
            Dict[str, torch.Tensor]: The dictionary of aggregated losses.
        """

        if loss_type == LossType.BACKWARD:
            loss_list = [losses.backward for losses in losses_list]
        else:
            loss_list = [losses.additional_losses for losses in losses_list]

        # We don't know the keys of the dict (backward or additional losses) beforehand. We obtain them
        # from the first entry because we know all of the losses will have the same keys
        loss_keys = loss_list[0].keys()
        num_losses = len(loss_list)
        loss_dict: Dict[str, torch.Tensor] = {}
        for key in loss_keys:
            loss = torch.sum(torch.FloatTensor([loss[key] for loss in loss_list]))
            if loss_meter_type == LossMeterType.AVERAGE:
                loss = loss / num_losses
            loss_dict[key] = loss

        return loss_dict


class LossAverageMeter(LossMeter):
    def __init__(self) -> None:
        """
        A meter to store and aggregate losses via averaging.
        """
        self.losses_list: List[Losses] = []

    def update(self, losses: Losses) -> None:
        """
        Appends loss to list of losses.

        Args:
            losses (Losses): A losses object with checkpoint, backward and additional losses.
        """
        self.losses_list.append(losses)

    def clear(self) -> None:
        """
        Reset the meter by re-initializing losses_list to be empty
        """
        self.losses_list = []

    def compute(self) -> Losses:
        """
        Compute average of current list of losses if non-empty.

        Returns:
            Losses: New Losses object with average of losses in losses_list.
        """
        assert len(self.losses_list) > 0

        num_losses = len(self.losses_list)
        # Compute average checkpoint and backward losses across list
        checkpoint_loss = torch.sum(torch.FloatTensor([losses.checkpoint for losses in self.losses_list])) / num_losses
        backward_loss = LossMeter.aggregate_loss_by_type(
            self.losses_list, loss_type=LossType.BACKWARD, loss_meter_type=LossMeterType.AVERAGE
        )
        additional_loss = LossMeter.aggregate_loss_by_type(
            self.losses_list, loss_type=LossType.ADDITIONAL, loss_meter_type=LossMeterType.AVERAGE
        )
        losses = Losses(backward=backward_loss, checkpoint=checkpoint_loss, additional_losses=additional_loss)
        return losses


class LossAccumulationMeter(LossMeter):
    def __init__(self) -> None:
        """
        A meter to store and aggregate losses via accumulation.
        """
        self.losses_list: List[Losses] = []

    def update(self, losses: Losses) -> None:
        """
        Appends loss to list of losses.

        Args:
            losses (Losses): A losses object with checkpoint, backward and additional losses.
        """
        self.losses_list.append(losses)

    def clear(self) -> None:
        """
        Reset the meter by re-initializing losses_list to be empty
        """
        self.losses_list = []

    def compute(self) -> Losses:
        """
        Compute sum of current list of losses if non-empty.

        Returns:
            Losses: New Losses object with sum of losses in losses_list.
        """
        assert len(self.losses_list) > 0

        # Compute average checkpoint and backward losses across list
        checkpoint_loss = torch.sum(torch.FloatTensor([losses.checkpoint for losses in self.losses_list]))
        backward_loss = LossMeter.aggregate_loss_by_type(
            self.losses_list, loss_type=LossType.BACKWARD, loss_meter_type=LossMeterType.ACCUMULATION
        )
        additional_loss = LossMeter.aggregate_loss_by_type(
            self.losses_list, loss_type=LossType.ADDITIONAL, loss_meter_type=LossMeterType.ACCUMULATION
        )
        losses = Losses(backward=backward_loss, checkpoint=checkpoint_loss, additional_losses=additional_loss)
        return losses
