from typing import Any, Optional

import torch
import torchmetrics
from flwr.common.typing import Metrics
from picai_eval import evaluate

from fl4health.utils.metrics import Metric


class nnUNetMetric(Metric):
    def __init__(self, metric: Metric):
        super().__init__(name=metric.name)
        self.metric = metric

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        # nnUNet models output seperate channels for each class (even in binary segmentation)
        # However labels are not onehot encoded to match leading to a dimension error when using torchmetrics
        # We only have a single target
        if pred.ndim != target.ndim:  # Add channel dimension if there isn't one
            target = target.view(target.shape[0], 1, *target.shape[1:])

        if pred.shape != target.shape:
            target_one_hot = torch.zeros(pred.shape, device=pred.device, dtype=torch.bool)
            target_one_hot.scatter(
                1, target.long(), 1
            )  # This does the onehot encoding. Its a weird function that is not intuitive

        self.metric.update(pred.long(), target_one_hot)

    def compute(self, name: Optional[str]) -> Metrics:
        return self.metric.compute(name=name)

    def clear(self) -> None:
        self.metric.clear()


class PICAI_AUROC(torchmetrics.Metric):
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


class PICAI_Score(torchmetrics.Metric):
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
