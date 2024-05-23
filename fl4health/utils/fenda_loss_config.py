from typing import Optional

import torch

from fl4health.losses.contrastive_loss import ContrastiveLoss
from fl4health.losses.cosine_similarity_loss import CosineSimilarityLoss
from fl4health.losses.perfcl_loss import PerFclLoss


class PerFclLossConfig:
    def __init__(
        self,
        device: torch.device,
        global_feature_contrastive_loss_weight: float,
        local_feature_contrastive_loss_weight: float,
        global_feature_loss_temperature: float = 0.5,
        local_feature_loss_temperature: float = 0.5,
    ) -> None:
        self.global_feature_contrastive_loss_weight = global_feature_contrastive_loss_weight
        self.local_feature_contrastive_loss_weight = local_feature_contrastive_loss_weight
        self.perfcl_loss_function = PerFclLoss(device, global_feature_loss_temperature, local_feature_loss_temperature)


class CosSimLossConfig:
    def __init__(self, device: torch.device, cos_sim_loss_weight: float) -> None:
        self.cos_sim_loss_weight = cos_sim_loss_weight
        self.cos_sim_loss_function = CosineSimilarityLoss(device)


class ContrastiveLossConfig:
    def __init__(self, device: torch.device, contrastive_loss_weight: float, temperature: float = 0.5) -> None:
        self.contrastive_loss_weight = contrastive_loss_weight
        self.contrastive_loss_function = ContrastiveLoss(device, temperature)


class ConstrainedFendaLossConfig:
    def __init__(
        self,
        perfcl_loss_config: Optional[PerFclLossConfig],
        cos_sim_loss_config: Optional[CosSimLossConfig],
        contrastive_loss_config: Optional[ContrastiveLossConfig],
    ) -> None:
        self.perfcl_loss_config = perfcl_loss_config
        self.cos_sim_loss_config = cos_sim_loss_config
        self.contrastive_loss_config = contrastive_loss_config

    def has_perfcl_loss(self) -> bool:
        return self.perfcl_loss_config is not None

    def has_cos_sim_loss(self) -> bool:
        return self.cos_sim_loss_config is not None

    def has_contrastive_loss(self) -> bool:
        return self.contrastive_loss_config is not None
