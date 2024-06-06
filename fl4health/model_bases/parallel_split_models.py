from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Tuple

import torch
import torch.nn as nn


class ParallelFeatureJoinMode(Enum):
    CONCATENATE = "CONCATENATE"
    SUM = "SUM"


class ParallelSplitHeadModule(nn.Module, ABC):
    def __init__(self, mode: ParallelFeatureJoinMode) -> None:
        """
        This is a head module to be used as part of ParallelSplitModel type models. This module is responsible for
        merging inputs from two parallel feature extractors and acting on those inputs to produce a prediction

        Args:
            mode (ParallelFeatureJoinMode): This determines HOW the head module is meant to combine the features
                produced by the extraction modules. Currently, there are two modes, concatenation or summation of the
                inputs before producing a prediction.
        """
        super().__init__()
        self.mode = mode

    @abstractmethod
    def parallel_output_join(self, local_tensor: torch.Tensor, global_tensor: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def head_forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, first_tensor: torch.Tensor, second_tensor: torch.Tensor) -> torch.Tensor:
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
        extractors are merged together and mapped to a prediction by a ParallelSplitHeadModule. By default, no
        feature tensors are stored. Only a prediction tensor is produced.

        Args:
            first_feature_extractor (nn.Module): First parallel feature extractor
            second_feature_extractor (nn.Module): Second parallel feature extractor
            model_head (ParallelSplitHeadModule): Module responsible for taking the outputs of the two feature
                extractors and using them to produce a prediction.
        """
        super().__init__()
        self.first_feature_extractor = first_feature_extractor
        self.second_feature_extractor = second_feature_extractor
        self.model_head = model_head

    def forward(self, input: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        first_output = self.first_feature_extractor.forward(input)
        second_output = self.second_feature_extractor.forward(input)
        preds = {"prediction": self.model_head.forward(first_output, second_output)}
        # No features are returned in the vanilla ParallelSplitModel implementation
        return preds, {}
