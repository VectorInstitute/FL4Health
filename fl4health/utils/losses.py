from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Generic, List, Optional, TypeVar, Union

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
    def aggregate(loss_meter: LossMeter[EvaluationLosses]) -> EvaluationLosses:
        """
        Aggregates the losses in the given LossMeter into an instance of EvaluationLosses

        Args:
            loss_meter (LossMeter[EvaluationLosses]): The loss meter object with the collected evaluation losses.

        Returns:
            EvaluationLosses: An instance of EvaluationLosses with the aggregated losses.
        """
        checkpoint_loss = torch.sum(
            torch.FloatTensor([losses.checkpoint for losses in loss_meter.losses_list])  # type: ignore
        )
        if loss_meter.loss_meter_type == LossMeterType.AVERAGE:
            checkpoint_loss /= len(loss_meter.losses_list)

        additional_losses_list = [losses.additional_losses for losses in loss_meter.losses_list]
        additional_losses_dict = LossMeter.aggregate_losses_dict(additional_losses_list, loss_meter.loss_meter_type)

        return EvaluationLosses(checkpoint=checkpoint_loss, additional_losses=additional_losses_dict)


class TrainingLosses(Losses):
    def __init__(
        self,
        backward: Union[torch.Tensor, Dict[str, torch.Tensor]],
        additional_losses: Optional[Dict[str, torch.Tensor]] = None,
    ) -> None:
        """
        A class to store the backward and additional_losses of a model
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
    def aggregate(loss_meter: LossMeter[TrainingLosses]) -> TrainingLosses:
        """
        Aggregates the losses in the given LossMeter into an instance of TrainingLosses

        Args:
            loss_meter (LossMeter[TrainingLosses]): The loss meter object with the collected training losses.

        Returns:
            TrainingLosses: An instance of TrainingLosses with the aggregated losses.
        """
        additional_losses_list = [losses.additional_losses for losses in loss_meter.losses_list]
        additional_losses_dict = LossMeter.aggregate_losses_dict(additional_losses_list, loss_meter.loss_meter_type)

        backward_losses_list = [losses.backward for losses in loss_meter.losses_list]  # type: ignore
        if len(backward_losses_list) > 0 and isinstance(backward_losses_list[0], dict):
            # if backward losses is a dictionary, aggregate the dictionary keys
            backward_losses_dict = LossMeter.aggregate_losses_dict(backward_losses_list, loss_meter.loss_meter_type)
            return TrainingLosses(backward=backward_losses_dict, additional_losses=additional_losses_dict)

        # otherwise, calculate the average tensor
        backward_losses = torch.sum(torch.FloatTensor(backward_losses_list))
        if loss_meter.loss_meter_type == LossMeterType.AVERAGE:
            backward_losses /= len(loss_meter.losses_list)

        return TrainingLosses(backward=backward_losses, additional_losses=additional_losses_dict)


class LossMeterType(Enum):
    AVERAGE = "AVERAGE"
    ACCUMULATION = "ACCUMULATION"


LossesType = TypeVar("LossesType", bound=Losses)


class LossMeter(Generic[LossesType]):
    def __init__(self, loss_meter_type: LossMeterType, losses_type: type[LossesType]) -> None:
        """
        A meter to store a list of losses.

        Args:
            loss_meter_type (LossMeterType): The type of this loss meter
            losses_type (type[Losses]): The type of the loss that will be stored. Should be one
                of the subclasses of Losses

        """
        self.losses_list: List[LossesType] = []
        self.loss_meter_type = loss_meter_type
        self.losses_type = losses_type

    def update(self, losses: LossesType) -> None:
        """
        Appends loss to list of losses.

        Args:
            losses (LossesType): A losses object with checkpoint, backward and additional losses.
        """
        self.losses_list.append(losses)

    def clear(self) -> None:
        """
        Resets the meter by re-initializing losses_list to be empty
        """
        self.losses_list = []

    def compute(self) -> LossesType:
        """
        Computes the aggregation of current list of losses if non-empty.

        Returns:
            LossesType: New Losses object with the aggregation of losses in losses_list.
        """
        assert len(self.losses_list) > 0
        return self.losses_type.aggregate(self)  # type: ignore

    @staticmethod
    def aggregate_losses_dict(
        loss_list: List[Dict[str, torch.Tensor]],
        loss_meter_type: LossMeterType,
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregates a list of losses dictionaries into a single dictionary according to the loss meter aggregation type

        Args:
            loss_list (List[Dict[str, torch.Tensor]]): A list of loss dictionaries
            loss_meter_type (LossMeterType): The type of the loss meter to perform the aggregation

        Returns:
            Dict[str, torch.Tensor]: A single dictionary with the aggregated losses according to the given loss
                meter type
        """
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
