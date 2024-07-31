from typing import Tuple

import torch
import torch.nn as nn

from fl4health.losses.contrastive_loss import MoonContrastiveLoss


class PerFclLoss(nn.Module):
    def __init__(
        self,
        device: torch.device,
        global_feature_loss_temperature: float = 0.5,
        local_feature_loss_temperature: float = 0.5,
    ) -> None:
        super().__init__()
        self.global_feature_contrastive_loss = MoonContrastiveLoss(device, global_feature_loss_temperature)
        self.local_feature_contrastive_loss = MoonContrastiveLoss(device, local_feature_loss_temperature)

    def forward(
        self,
        local_features: torch.Tensor,
        old_local_features: torch.Tensor,
        global_features: torch.Tensor,
        old_global_features: torch.Tensor,
        initial_global_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        PerFCL loss implemented based on https://www.sciencedirect.com/science/article/pii/S0031320323002078. This
        paper introduced two contrastive loss functions:
            1 - First one aims to enhance the similarity between the current global features (z_s) and aggregated
                global features (z_g) (saved at the start of client-side training) as positive pairs while reducing
                the similarity between the current global features (z_s) and old global features (hat{z}_s) from the
                end of the previous client-side training as negative pairs.
            2- Second one aims to enhance the similarity between the current local features (z_p) and old local
                features (hat{z}_p) from the end of the previous client-side training as positive pairs while
                reducing the similarity between the current local features (z_p) and aggregated global features (z_g)
                (saved at the start of client-side training) as negative pairs.
        Args:
            local_features (torch.Tensor): Features produced by the local feature extractor of the model during the
                client-side training. Denoted as z_p in the original paper. Shape (batch_size, n_features)
            old_local_features (torch.Tensor): Features produced by the FINAL local feature extractor of the model
                from the PREVIOUS server round. Denoted as hat{z}_p in the original paper.
                Shape (batch_size, n_features)
            global_features (torch.Tensor): Features produced by the global feature extractor of the model during the
                client-side training. Denoted as z_s in the original paper. Shape (batch_size, n_features)
            old_global_features (torch.Tensor): Features produced by the FINAL global feature extractor of the model
                from the PREVIOUS server round. Denoted as hat{z}_s in the original paper.
                Shape (batch_size, n_features)
            initial_global_features (torch.Tensor): Features produced by the INITIAL global feature extractor of the
                model at the start of client-side training. This feature extractor is the AGGREGATED weights across
                clients. Shape (batch_size, n_features)
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple containing the two components of the PerFCL loss function to
                be weighted and summed. The first tensor corresponds to the global feature loss, the second is
                associated with the local feature loss.
        """
        # The more general contrastive loss function requires 3D tensors, where the first dimension is potentially
        # greater than 1. For the PerFCL loss, this will always just be 1.
        old_local_features = old_local_features.unsqueeze(0)
        old_global_features = old_global_features.unsqueeze(0)
        initial_global_features = initial_global_features.unsqueeze(0)

        global_feature_loss = self.global_feature_contrastive_loss(
            features=global_features,  # (z_s)
            positive_pairs=initial_global_features,  # (z_g)
            negative_pairs=old_global_features,  # (\hat{z_s})
        )
        local_feature_loss = self.local_feature_contrastive_loss(
            features=local_features,  # (z_p)
            positive_pairs=old_local_features,  # (\hat{z_p})
            negative_pairs=initial_global_features,  # (z_g)
        )

        return global_feature_loss, local_feature_loss
