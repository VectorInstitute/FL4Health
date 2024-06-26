from typing import Any

import torch
from picai_eval import evaluate
from torchmetrics import Metric


class PICAI_AUROC(Metric):
    def __init__(self, **kwargs: Any) -> None:
        """
        Area Under the Reciever Operating Characteristic curve for PICAI. This \
            is a patient level metric meaning that the model is evaluated on its\
            ability to detect whether or not a patient has a tumour and not on \
            the quality of it's segmentation. Detection requires a minimum \
            overlap between the ground truth tumour and the segmentation map \
            however
        """
        super().__init__(**kwargs)
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")

    def update(self, input: torch.Tensor, target: torch.Tensor) -> None:
        self.preds.append(input)
        self.targets.append(target)

    def compute(self) -> Any:
        metrics = evaluate(y_det=self.preds, y_true=self.targets)
        return metrics.auroc


class PICAI_Score(Metric):
    def __init__(self, **kwargs: Any) -> None:
        """
        The Picai score is the average of the patient level AUROC and the \
        segmentation level Average Precision
        """
        super().__init__(**kwargs)
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")

    def update(self, input: torch.Tensor, target: torch.Tensor) -> None:
        self.preds.append(input)
        self.targets.append(target)

    def compute(self) -> Any:
        metrics = evaluate(y_det=self.preds, y_true=self.targets)
        return metrics.score
