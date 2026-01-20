from abc import ABC, abstractmethod
from enum import Enum

import torch
from torch import nn


class ParallelFeatureJoinMode(Enum):
    CONCATENATE = "CONCATENATE"
    SUM = "SUM"


class ParallelSplitHeadModule(nn.Module, ABC):
    def __init__(self, mode: ParallelFeatureJoinMode) -> None:
        """
        This is a head module to be used as part of ``ParallelSplitModel`` type models. This module is responsible for
        merging inputs from two parallel feature extractors and acting on those inputs to produce a prediction.

        Args:
            mode (ParallelFeatureJoinMode): This determines **HOW** the head module is meant to combine the features
                produced by the extraction modules. Currently, there are two modes, concatenation or summation of the
                inputs before producing a prediction.
        """
        super().__init__()
        self.mode = mode

    @abstractmethod
    def parallel_output_join(self, local_tensor: torch.Tensor, global_tensor: torch.Tensor) -> torch.Tensor:
        """
        Defines how the local and global feature tensors that are output by the preceding parallel feature extractors
        are meant to be joined together when the ``self.mode`` is set to ``ParallelFeatureJoinMode.CONCATENATE``.

        Args:
            local_tensor (torch.Tensor): First tensor to be joined.
            global_tensor (torch.Tensor): Second tensor to be joined.

        Raises:
            NotImplementedError: Any implementing child class must produce this method if it is to be used.

        Returns:
            (torch.Tensor): A single tensor with the two tensors joined together in some way.
        """
        raise NotImplementedError

    @abstractmethod
    def head_forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward function for the head module.

        Args:
            input_tensor (torch.Tensor): Input tensor to be mapped.

        Raises:
            NotImplementedError: Must be implemented by any child class.

        Returns:
            (torch.Tensor): Output of the head module from the given input.
        """
        raise NotImplementedError

    def forward(self, first_tensor: torch.Tensor, second_tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward function for the head module of ``ParallelSplitModels``. The inputs (``first_tensor``,
        ``second_tensor``) are concatenated or added together depending on the mode specified in ``self.mode``
        The concatenation procedure is defined by ``parallel_output_join``. This concatenated or added together
        tensor is then passed through the forward function of the head module.

        Args:
            first_tensor (torch.Tensor): Output from one parallel module.
            second_tensor (torch.Tensor): Output from one parallel module.

        Returns:
            (torch.Tensor): Output from the head module.
        """
        head_input = (
            self.parallel_output_join(first_tensor, second_tensor)
            if self.mode == ParallelFeatureJoinMode.CONCATENATE
            else torch.add(first_tensor, second_tensor)
        )
        return self.head_forward(head_input)


class ParallelSplitModel(nn.Module):
    def __init__(
        self,
        first_feature_extractor: nn.Module,
        second_feature_extractor: nn.Module,
        model_head: ParallelSplitHeadModule,
    ) -> None:
        """
        This defines a model that has been split into two parallel feature extractors. The outputs of these feature
        extractors are merged together and mapped to a prediction by a ``ParallelSplitHeadModule``. By default, no
        feature tensors are stored. Only a prediction tensor is produced.

        Args:
            first_feature_extractor (nn.Module): First parallel feature extractor.
            second_feature_extractor (nn.Module): Second parallel feature extractor.
            model_head (ParallelSplitHeadModule): Module responsible for taking the outputs of the two feature
                extractors and using them to produce a prediction.
        """
        super().__init__()
        self.first_feature_extractor = first_feature_extractor
        self.second_feature_extractor = second_feature_extractor
        self.model_head = model_head

    def forward(self, input: torch.Tensor) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """
        Composite forward function. The input tensor is first passed through the two parallel feature extractors and
        then finally through the head model. The outputs and joining mechanism defined in the head model need to
        be compatible with the head model input itself. This is left to the user to handle.

        Args:
            input (torch.Tensor): Input tensor to be passed through the set of forwards.

        Returns:
            (tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]): Prediction tensor from the head model. These
                predictions are stored under the "prediction" key of the dictionary. The second feature dictionary is
                empty by default.
        """
        first_output = self.first_feature_extractor.forward(input)
        second_output = self.second_feature_extractor.forward(input)
        preds = {"prediction": self.model_head.forward(first_output, second_output)}
        # No features are returned in the vanilla ParallelSplitModel implementation
        return preds, {}
