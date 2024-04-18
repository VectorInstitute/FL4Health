from enum import Enum
from typing import Dict, List, Optional

import torch
import torch.nn as nn


class EnsembleAggregationMode(Enum):
    VOTE = "VOTE"
    AVERAGE = "AVERAGE"


class EnsembleModel(nn.Module):
    def __init__(
        self,
        ensemble_models: Dict[str, nn.Module],
        aggregation_mode: Optional[EnsembleAggregationMode] = EnsembleAggregationMode.AVERAGE,
    ) -> None:
        """
        Class that acts a wrapper to an ensemble of models to be trained in federated manner with support
        for both voting and averaging prediction of individual models.

        Args:
            ensemble_models (Dict[str, nn.Module]): A dictionary of models that make up the ensemble.
            aggregation_mode (Optional[EnsembleAggregationMode]): The mode in which to aggregate the
                predictions of individual models.
        """
        super().__init__()

        self.ensemble_models = nn.ModuleDict(ensemble_models)
        self.aggregation_mode = aggregation_mode

    def forward(self, input: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Produce the predictions of the ensemble models given input data.

        Args:
            input (torch.Tensor): A batch of input data.

        Returns:
            Dict[str, torch.Tensor]: A dictionary of predictions of the individual ensemble models
                as well as prediction of the ensemble as a whole.
        """
        preds = {}
        for key, model in self.ensemble_models.items():
            preds[key] = model(input)

        # Don't store gradients when computing ensemble predictions
        with torch.no_grad():
            if self.aggregation_mode == EnsembleAggregationMode.AVERAGE:
                ensemble_pred = self.ensemble_average(list(preds.values()))
            else:
                ensemble_pred = self.ensemble_vote(list(preds.values()))

        preds["ensemble-pred"] = ensemble_pred

        return preds

    def ensemble_vote(self, preds_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Produces the aggregated prediction of the ensemble via voting. Expects predictions
        to be in a format where the 0 axis represents the sample index and the -1 axis represents
        the class dimension.

        Args:
            preds_list (List[torch.Tensor]): A list of predictions of the models in the ensemble.

        Returns:
            torch.Tensor: The vote prediction of the ensemble.
        """
        assert all(preds.shape == preds_list[0].shape for preds in preds_list)
        preds_dimension = list(preds_list[0].shape)

        # If larger than two dimensions, we map to 2D to perform voting operation (and reshape later)
        if len(preds_dimension) > 2:
            preds_list = [preds.reshape(-1, preds_dimension[-1]) for preds in preds_list]

        # For each model prediction, compute the argmax of the model over the classes and stack column-wise into matrix
        # Each row of matrix represents the argmax of each model for a given sample
        argmax_per_model = torch.hstack([torch.argmax(preds, dim=1, keepdim=True) for preds in preds_list])
        # For each row (sample), compute the unique class predictions and their respective counts
        index_count_list = map(lambda x: torch.unique(x, return_counts=True), argmax_per_model.unbind())
        # For each element of list (class index, class count) pairing
        # extract index with the highest count and create tensor
        indices_with_highest_counts = torch.tensor([index[torch.argmax(count)] for index, count in index_count_list])
        # One hot encode ensemble prediction for each sample
        vote_preds = nn.functional.one_hot(indices_with_highest_counts, num_classes=preds_dimension[-1])

        # If larger than two dimensions, map back to original dimensions
        if len(preds_dimension) > 2:
            vote_preds = vote_preds.reshape(*preds_dimension)

        return vote_preds

    def ensemble_average(self, preds_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Produces the aggregated prediction of the ensemble via averaging.

        Args:
            preds_list (List[torch.Tensor]): A list of predictions of the models in the ensemble.

        Returns:
            torch.Tensor: The average prediction of the ensemble.
        """
        stacked_model_preds = torch.stack(preds_list)
        avg_preds = torch.mean(stacked_model_preds, dim=0)
        return avg_preds
