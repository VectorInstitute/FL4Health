from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Optional

import torch


class Losses:
    def __init__(
        self,
        checkpoint: torch.Tensor,
        backward: torch.Tensor,
        additional_losses: Optional[Dict[str, torch.Tensor]] = None,
    ):
        self.checkpoint = checkpoint
        self.backward = backward
        self.additional_losses = additional_losses if additional_losses else {}

    def as_dict(self) -> Dict[str, float]:
        loss_dict: Dict[str, float] = {}
        loss_dict["checkpoint"] = float(self.checkpoint.item())
        loss_dict["backward"] = float(self.backward.item())

        if self.additional_losses is not None:
            for key, val in self.additional_losses.items():
                loss_dict[key] = float(val.item())

        return loss_dict


class LossMeterType(Enum):
    AVERAGE = "AVERAGE"
    ACCUMULATION = "ACCUMULATION"


class LossMeter(ABC):
    @abstractmethod
    def update(self, losses: Losses) -> None:
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def compute(self) -> Losses:
        raise NotImplementedError

    @classmethod
    def get_meter_by_type(cls, meter_enum: LossMeterType) -> LossMeter:
        if meter_enum == LossMeterType.AVERAGE:
            return LossAverageMeter()
        elif meter_enum == LossMeterType.ACCUMULATION:
            return LossAccumulationMeter()
        else:
            raise ValueError(f"Not supported meter type: {str(meter_enum)}")


class LossAverageMeter(LossMeter):
    def __init__(self) -> None:
        self.losses_list: List[Losses] = []

    def update(self, losses: Losses) -> None:
        self.losses_list.append(losses)

    def clear(self) -> None:
        self.losses_list = []

    def compute(self) -> Losses:
        assert len(self.losses_list) > 0

        num_losses = len(self.losses_list)
        # Compute average checkpoint and backward losses across list
        checkpoint_loss = torch.sum(torch.FloatTensor([losses.checkpoint for losses in self.losses_list])) / num_losses
        backward_loss = torch.sum(torch.FloatTensor([losses.backward for losses in self.losses_list])) / num_losses

        # We don't know the keys of the additional_losses beforehand so we extract them from the first entry
        # because we know all of the losses will have the same keys in additinal_losses dict
        additional_losses: Dict[str, torch.Tensor] = {}
        for key in self.losses_list[0].additional_losses.keys():
            additional_losses[key] = (
                torch.sum(torch.FloatTensor([losses.additional_losses[key] for losses in self.losses_list]))
                / num_losses
            )

        losses = Losses(checkpoint=checkpoint_loss, backward=backward_loss, additional_losses=additional_losses)
        return losses


class LossAccumulationMeter(LossMeter):
    def __init__(self) -> None:
        self.losses_list: List[Losses] = []

    def update(self, losses: Losses) -> None:
        self.losses_list.append(losses)

    def clear(self) -> None:
        self.losses_list = []

    def compute(self) -> Losses:
        assert len(self.losses_list) > 0

        # Compute average checkpoint and backward losses across list
        checkpoint_loss = torch.sum(torch.FloatTensor([losses.checkpoint for losses in self.losses_list]))
        backward_loss = torch.sum(torch.FloatTensor([losses.backward for losses in self.losses_list]))

        # We don't know the keys of the additional_losses beforehand so we extract them from the first entry
        # because we know all of the losses will have the same keys in additinal_losses dict
        additional_losses: Dict[str, torch.Tensor] = {}
        for key in self.losses_list[0].additional_losses.keys():
            additional_losses[key] = torch.sum(
                torch.FloatTensor([losses.additional_losses[key] for losses in self.losses_list])
            )

        losses = Losses(checkpoint=checkpoint_loss, backward=backward_loss, additional_losses=additional_losses)
        return losses


class SupConLoss(torch.nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature: float = 0.07, contrast_mode: str = "all", base_temperature: float = 0.07) -> None:
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(
        self, features: torch.Tensor, labels: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = torch.device("cuda") if features.is_cuda else torch.device("cpu")

        if len(features.shape) < 3:
            raise ValueError("`features` needs to be [bsz, n_views, ...]," "at least 3 dimensions are required")
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if mask is None:
            if labels is None:
                mask = torch.eye(batch_size, dtype=torch.float32).to(device)
            else:
                labels = labels.contiguous().view(-1, 1)
                if labels.shape[0] != batch_size:
                    raise ValueError("Num of labels does not match num of features")
                mask = torch.eq(labels, labels.T).float().to(device)
        else:
            if labels is None:
                mask = mask.float().to(device)
            else:
                raise ValueError("Cannot define both `labels` and `mask`")

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == "one":
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == "all":
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError("Unknown mode: {}".format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask), 1, torch.arange(batch_size * anchor_count).view(-1, 1).to(device), 0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask + 1e-10
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
