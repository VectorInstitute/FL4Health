from enum import Enum
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn


class EnsembleAggregationMode(Enum):
    VOTE = "VOTE"
    AVERAGE = "AVERAGE"


class EnsembleModel(nn.Module):
    def __init__(
        self,
        models: Sequence[nn.Module],
        aggregation_mode: Optional[EnsembleAggregationMode] = EnsembleAggregationMode.AVERAGE,
    ) -> None:
        super().__init__()
        self.models = models
        self.aggregation_mode = aggregation_mode

    def forward(self, input: torch.Tensor) -> Dict[str, torch.Tensor]:
        preds = {}
        for i, model in enumerate(self.models):
            preds[f"ensemble-model-{str(i)}"] = model(input).squeeze()

        if self.aggregation_mode == EnsembleAggregationMode.AVERAGE:
            ensemble_pred = self.ensemble_average(preds)
        else:
            ensemble_pred = self.ensemble_vote(preds)

        preds["ensemble-pred"] = ensemble_pred

        return preds

    def ensemble_vote(self, model_preds: Dict[str, torch.Tensor]) -> torch.Tensor:
        argmax_per_model = torch.stack([torch.argmax(val, dim=1) for val in model_preds.values()])
        argmax = torch.max(argmax_per_model, dim=0)[0]
        vote_preds = nn.functional.one_hot(argmax, num_classes=model_preds["ensemble-model-0"].shape[-1])
        return vote_preds

    def ensemble_average(self, model_preds: Dict[str, torch.Tensor]) -> torch.Tensor:
        stacked_model_preds = torch.stack(list(model_preds.values()))
        avg_preds = torch.mean(stacked_model_preds, dim=0)
        return avg_preds
