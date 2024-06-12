from abc import ABC, abstractmethod
from typing import Dict

import torch
import torch.nn as nn


class FederatedSelfSupervisedModel(ABC, nn.Module):
    def __init__(
        self,
        model: nn.Module,
    ) -> None:
        self.model = model

    @abstractmethod
    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError


class FedSimClr(FederatedSelfSupervisedModel):
    def __init__(self, model: nn.Module, projection_head: nn.Module = nn.Identity()) -> None:
        super().__init__(model)
        self.projection_head = projection_head

    def encode(self, input: torch.Tensor) -> torch.Tensor:
        features = self.model(input)
        return self.projection_head(features)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        assert "input" in inputs and "transformed_input" in inputs

        features = self.encode(inputs["input"])
        transformed_features = self.encode(inputs["transformed_input"])

        return {"features": features, "transformed_features": transformed_features}
