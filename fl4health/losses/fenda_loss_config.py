from typing import Optional, Tuple

import torch

from fl4health.losses.contrastive_loss import MoonContrastiveLoss
from fl4health.losses.cosine_similarity_loss import CosineSimilarityLoss
from fl4health.losses.perfcl_loss import PerFclLoss


class PerFclLossContainer:
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


class CosineSimilarityLossContainer:
    def __init__(self, device: torch.device, cos_sim_loss_weight: float) -> None:
        self.cos_sim_loss_weight = cos_sim_loss_weight
        self.cos_sim_loss_function = CosineSimilarityLoss(device)


class MoonContrastiveLossContainer:
    def __init__(self, device: torch.device, contrastive_loss_weight: float, temperature: float = 0.5) -> None:
        self.contrastive_loss_weight = contrastive_loss_weight
        self.contrastive_loss_function = MoonContrastiveLoss(device, temperature)


class ConstrainedFendaLossContainer:
    def __init__(
        self,
        perfcl_loss_config: Optional[PerFclLossContainer],
        cosine_similarity_loss_config: Optional[CosineSimilarityLossContainer],
        contrastive_loss_config: Optional[MoonContrastiveLossContainer],
    ) -> None:
        self.perfcl_loss_config = perfcl_loss_config
        self.cos_sim_loss_config = cosine_similarity_loss_config
        self.contrastive_loss_config = contrastive_loss_config

    def has_perfcl_loss(self) -> bool:
        return self.perfcl_loss_config is not None

    def has_cosine_similarity_loss(self) -> bool:
        return self.cos_sim_loss_config is not None

    def has_contrastive_loss(self) -> bool:
        return self.contrastive_loss_config is not None

    def compute_contrastive_loss(
        self, features: torch.Tensor, positive_pairs: torch.Tensor, negative_pairs: torch.Tensor
    ) -> torch.Tensor:
        assert self.contrastive_loss_config is not None
        contrastive_loss = self.contrastive_loss_config.contrastive_loss_function(
            features, positive_pairs, negative_pairs
        )
        return self.contrastive_loss_config.contrastive_loss_weight * contrastive_loss

    def compute_cosine_similarity_loss(
        self, first_features: torch.Tensor, second_features: torch.Tensor
    ) -> torch.Tensor:
        assert self.cos_sim_loss_config is not None
        cosine_similarity_loss = self.cos_sim_loss_config.cos_sim_loss_function(first_features, second_features)
        return self.cos_sim_loss_config.cos_sim_loss_weight * cosine_similarity_loss

    def compute_perfcl_loss(
        self,
        local_features: torch.Tensor,
        old_local_features: torch.Tensor,
        global_features: torch.Tensor,
        old_global_features: torch.Tensor,
        initial_global_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert self.perfcl_loss_config is not None
        global_feature_contrastive_loss, local_feature_contrastive_loss = self.perfcl_loss_config.perfcl_loss_function(
            local_features, old_local_features, global_features, old_global_features, initial_global_features
        )
        return (
            self.perfcl_loss_config.global_feature_contrastive_loss_weight * global_feature_contrastive_loss,
            self.perfcl_loss_config.local_feature_contrastive_loss_weight * local_feature_contrastive_loss,
        )
