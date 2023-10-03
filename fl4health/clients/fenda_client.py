from pathlib import Path
from typing import Optional, Sequence

import torch
from flwr.common.typing import Config

from fl4health.checkpointing.checkpointer import TorchCheckpointer
from fl4health.clients.basic_client import BasicClient
from fl4health.model_bases.fenda_base import FendaModel
from fl4health.parameter_exchange.layer_exchanger import FixedLayerExchanger
from fl4health.parameter_exchange.parameter_exchanger_base import ParameterExchanger
from fl4health.utils.losses import Losses, LossMeterType
from fl4health.utils.metrics import Metric, MetricMeterType


class FendaClient(BasicClient):
    def __init__(
        self,
        data_path: Path,
        metrics: Sequence[Metric],
        device: torch.device,
        loss_meter_type: LossMeterType = LossMeterType.AVERAGE,
        metric_meter_type: MetricMeterType = MetricMeterType.AVERAGE,
        use_wandb_reporter: bool = False,
        checkpointer: Optional[TorchCheckpointer] = None,
    ) -> None:
        super().__init__(
            data_path, metrics, device, loss_meter_type, metric_meter_type, use_wandb_reporter, checkpointer
        )
        self.cos_sim = torch.nn.CosineSimilarity(dim=0)
        self.contrastive = SupConLoss()
        self.local_features: torch.Tensor
        self.global_features: torch.Tensor

    def get_parameter_exchanger(self, config: Config) -> ParameterExchanger:
        assert isinstance(self.model, FendaModel)
        return FixedLayerExchanger(self.model.layers_to_exchange())

    def predict(self, input: torch.Tensor) -> torch.Tensor:
        pred, self.local_features, self.global_features = self.model(input, self.pre_train)
        return pred

    def get_cosine_similarity_loss(self) -> torch.Tensor:

        assert len(self.local_features) == len(self.global_features)

        return self.cos_sim(self.local_features, self.global_features).mean()

    def get_contrastive_loss(self) -> torch.Tensor:
        assert len(self.local_features) == len(self.global_features)
        labels = torch.cat((torch.ones(1), torch.zeros(1))).to(self.device)
        return self.contrastive(
            torch.cat((self.local_features.unsqueeze(0), self.global_features.unsqueeze(0)), dim=0), labels=labels
        )

    def compute_loss(self, preds: torch.Tensor, target: torch.Tensor) -> Losses:
        if self.pre_train:
            return super().compute_loss(preds, target)
        loss = self.criterion(preds, target)
        # cos_loss = self.get_cosine_similarity_loss()
        # total_loss = loss + cos_loss
        # losses = Losses(checkpoint=loss, backward=total_loss, additional_losses={"cos_sim_loss": cos_loss})
        contrastive_loss = self.get_contrastive_loss()
        total_loss = loss + 0.1 * contrastive_loss
        losses = Losses(checkpoint=loss, backward=total_loss, additional_losses={"contrastive_loss": contrastive_loss})
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
