from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Optional, Union

import torch


class Losses(ABC):
    def __init__(self, additional_losses: Optional[Dict[str, torch.Tensor]] = None) -> None:
        """
        An abstract class to store the losses

        Args:
            additional_losses (Optional[Dict[str, torch.Tensor]]): Optional dictionary of additional losses.
        """
        self.additional_losses = additional_losses if additional_losses else {}

    def as_dict(self) -> Dict[str, float]:
        """
        Produces a dictionary representation of the object with all of the losses.

        Returns:
            Dict[str, float]: A dictionary with the additional losses if they exist.
        """
        loss_dict: Dict[str, float] = {}

        if self.additional_losses is not None:
            for key, val in self.additional_losses.items():
                loss_dict[key] = float(val.item())

        return loss_dict

    @staticmethod
    @abstractmethod
    def aggregate(loss_meter: LossMeter) -> Losses:
        """
        Aggregates the losses in the given LossMeter into an instance of Losses

        Args:
            loss_meter (LossMeter): The loss meter object with the collected losses.

        Raises:
            NotImplementedError: To be implemented by child classes.
        """
        raise NotImplementedError


class EvaluationLosses(Losses):
    def __init__(self, checkpoint: torch.Tensor, additional_losses: Optional[Dict[str, torch.Tensor]] = None) -> None:
        """
        A class to store the checkpoint and additional_losses of a model
        along with a method to return a dictionary representation.

        Args:
            checkpoint (torch.Tensor): The loss used to checkpoint model (if checkpointing is enabled).
            additional_losses (Optional[Dict[str, torch.Tensor]]): Optional dictionary of additional losses.
        """
        super().__init__(additional_losses)
        self.checkpoint = checkpoint

    def as_dict(self) -> Dict[str, float]:
        """
        Produces a dictionary representation of the object with all of the losses.

        Returns:
            Dict[str, float]: A dictionary with the checkpoint loss, plus each one of the keys in
                additional losses if they exist.
        """
        loss_dict = super().as_dict()
        loss_dict["checkpoint"] = float(self.checkpoint.item())
        return loss_dict

    @staticmethod
    def aggregate(loss_meter: LossMeter) -> EvaluationLosses:
        """
        Aggregates the losses in the given LossMeter into an instance of EvaluationLosses

        Args:
            loss_meter (LossMeter): The loss meter object with the collected losses.

        Returns:
            EvaluationLosses: An instance of EvaluationLosses with the aggregated losses.
        """
        assert all([isinstance(losses, EvaluationLosses) for losses in loss_meter.losses_list])
        checkpoint_loss = torch.sum(
            torch.FloatTensor([losses.checkpoint for losses in loss_meter.losses_list])  # type: ignore
        )
        if loss_meter.get_type() == LossMeterType.AVERAGE:
            checkpoint_loss /= len(loss_meter.losses_list)

        additional_losses_list = [losses.additional_losses for losses in loss_meter.losses_list]
        additional_losses_dict = _aggregate_losses_dict(additional_losses_list, loss_meter.get_type())

        return EvaluationLosses(checkpoint=checkpoint_loss, additional_losses=additional_losses_dict)


class TrainingLosses(Losses):
    def __init__(
        self,
        backward: Union[torch.Tensor, Dict[str, torch.Tensor]],
        additional_losses: Optional[Dict[str, torch.Tensor]] = None,
    ) -> None:
        """
        A class to store the checkpoint, backward and additional_losses of a model
        along with a method to return a dictionary representation.

        Args:
            backward (Union[torch.Tensor, Dict[str, torch.Tensor]]): The backward loss or
                losses to optimize. In the normal case, backward is a Tensor corresponding to the
                loss of a model. In the case of an ensemble_model, backward is dictionary of losses.
            additional_losses (Optional[Dict[str, torch.Tensor]]): Optional dictionary of additional losses.
        """
        super().__init__(additional_losses)
        self.backward = backward if isinstance(backward, dict) else {"backward": backward}

    def as_dict(self) -> Dict[str, float]:
        """
        Produces a dictionary representation of the object with all of the losses.

        Returns:
            Dict[str, float]: A dictionary where each key represents one of the  backward losses,
                plus additional losses if they exist.
        """
        loss_dict = super().as_dict()

        backward = {key: float(loss.item()) for key, loss in self.backward.items()}
        loss_dict.update(backward)

        return loss_dict

    @staticmethod
    def aggregate(loss_meter: LossMeter) -> TrainingLosses:
        """
        Aggregates the losses in the given LossMeter into an instance of TrainingLosses

        Args:
            loss_meter (LossMeter): The loss meter object with the collected losses.

        Returns:
            TrainingLosses: An instance of TrainingLosses with the aggregated losses.
        """
        assert all([isinstance(losses, TrainingLosses) for losses in loss_meter.losses_list])

        additional_losses_list = [losses.additional_losses for losses in loss_meter.losses_list]
        additional_losses_dict = _aggregate_losses_dict(additional_losses_list, loss_meter.get_type())

        backward_losses_list = [losses.backward for losses in loss_meter.losses_list]  # type: ignore
        if len(backward_losses_list) > 0 and isinstance(backward_losses_list[0], dict):
            # if backward losses is a dictionary, aggregate the dictionary keys
            backward_losses_dict = _aggregate_losses_dict(backward_losses_list, loss_meter.get_type())
            return TrainingLosses(backward=backward_losses_dict, additional_losses=additional_losses_dict)

        # otherwise, calculate the average tensor
        backward_losses = torch.sum(torch.FloatTensor(backward_losses_list))
        if loss_meter.get_type() == LossMeterType.AVERAGE:
            backward_losses /= len(loss_meter.losses_list)

        return TrainingLosses(backward=backward_losses, additional_losses=additional_losses_dict)


class LossMeterType(Enum):
    AVERAGE = "AVERAGE"
    ACCUMULATION = "ACCUMULATION"


class LossMeter(ABC):
    def __init__(self) -> None:
        """
        A meter to store a list of losses.
        """
        self.losses_list: List[Losses] = []

    def get_type(self) -> LossMeterType:
        """
        Returns the type of this loss meter

        Raises:
            NotImplementedError: To be implemented by child classes.
        """
        raise NotImplementedError

    @abstractmethod
    def update(self, losses: Losses) -> None:
        """
        Appends loss to list of losses.

        Args:
            losses (Losses): A losses object with checkpoint, backward and additional losses.

        Raises:
            NotImplementedError: To be implemented by child classes.
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

    def compute(self) -> Losses:
        """
        Compute average of current list of losses if non-empty.

        Returns:
            Losses: New Losses object with average of losses in losses_list.
        """
        assert len(self.losses_list) > 0

        # assume the type of the first loss object is the type of all loss objects
        if isinstance(self.losses_list[0], EvaluationLosses):
            return EvaluationLosses.aggregate(self)

        return TrainingLosses.aggregate(self)

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


class LossAverageMeter(LossMeter):
    def __init__(self) -> None:
        """
        A meter to store and aggregate losses via averaging.
        """
        super().__init__()

    def get_type(self) -> LossMeterType:
        """
        Returns the type of this loss meter

        Returns:
            LossMeterType.AVERAGE
        """
        return LossMeterType.AVERAGE

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


class LossAccumulationMeter(LossMeter):
    def __init__(self) -> None:
        """
        A meter to store and aggregate losses via accumulation.
        """
        super().__init__()

    def get_type(self) -> LossMeterType:
        """
        Returns the type of this loss meter

        Returns:
            LossMeterType.ACCUMULATION
        """
        return LossMeterType.ACCUMULATION

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


def _aggregate_losses_dict(
    loss_list: List[Dict[str, torch.Tensor]],
    loss_meter_type: LossMeterType,
) -> Dict[str, torch.Tensor]:
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
