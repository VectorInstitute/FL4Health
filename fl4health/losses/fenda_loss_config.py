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
        """
        Container to hold the different pieces associated with PerFCL Loss.

        Args:
            device (torch.device): Device to which the loss will be sent and computed on.
            global_feature_contrastive_loss_weight (float): Weight on the global contrastive loss function.
            local_feature_contrastive_loss_weight (float): Weight on the local model contrastive loss function.
            global_feature_loss_temperature (float, optional): Temperature parameter on the global contrastive loss
                function. Defaults to 0.5.
            local_feature_loss_temperature (float, optional): Temperature parameter on the local contrastive loss
                function. Defaults to 0.5.
        """
        self.global_feature_contrastive_loss_weight = global_feature_contrastive_loss_weight
        self.local_feature_contrastive_loss_weight = local_feature_contrastive_loss_weight
        self.perfcl_loss_function = PerFclLoss(device, global_feature_loss_temperature, local_feature_loss_temperature)


class CosineSimilarityLossContainer:
    def __init__(self, device: torch.device, cos_sim_loss_weight: float) -> None:
        """
        Container to hold the different pieces associated with cosine similarity.

        Args:
            device (torch.device): Device to which the loss will be sent and computed on.
            cos_sim_loss_weight (float): Weight associated with the cosine loss function in optimization.
        """
        self.cos_sim_loss_weight = cos_sim_loss_weight
        self.cos_sim_loss_function = CosineSimilarityLoss(device)


class MoonContrastiveLossContainer:
    def __init__(self, device: torch.device, contrastive_loss_weight: float, temperature: float = 0.5) -> None:
        """
        Container to hold the different pieces associated with Moon Contrastive loss function.

        Args:
            device (torch.device): Device to which the loss will be sent and computed on.
            contrastive_loss_weight (float): Weight associated with the contrastive loss function in optimization.
            temperature (float, optional): Temperature parameter on the contrastive loss function.
                Defaults to 0.5.
        """
        self.contrastive_loss_weight = contrastive_loss_weight
        self.contrastive_loss_function = MoonContrastiveLoss(device, temperature)


class ConstrainedFendaLossContainer:
    def __init__(
        self,
        perfcl_loss_config: PerFclLossContainer | None,
        cosine_similarity_loss_config: CosineSimilarityLossContainer | None,
        contrastive_loss_config: MoonContrastiveLossContainer | None,
    ) -> None:
        """
        Container to gather all of the possible loss functions used in constrained FENDA model optimization.

        Args:
            perfcl_loss_config (PerFclLossContainer | None): PerFCL loss container. If none, the loss isn not used.
            cosine_similarity_loss_config (CosineSimilarityLossContainer | None): Cosine similarity loss container.
                If none the loss is not used.
            contrastive_loss_config (MoonContrastiveLossContainer | None): Contrastive loss container. If none, the
                loss is not used.
        """
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
        """
        Compute the contrastive loss, if it exists, using the configuration.

        Args:
            features (torch.Tensor): features from the model.
            positive_pairs (torch.Tensor): positive pair features to compare to.
            negative_pairs (torch.Tensor): negative pair features to compare to.

        Returns:
            (torch.Tensor): loss function
        """
        assert self.contrastive_loss_config is not None
        contrastive_loss = self.contrastive_loss_config.contrastive_loss_function(
            features, positive_pairs, negative_pairs
        )
        return self.contrastive_loss_config.contrastive_loss_weight * contrastive_loss

    def compute_cosine_similarity_loss(
        self, first_features: torch.Tensor, second_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the cosine loss, if it exists, using the configuration.

        Args:
            first_features (torch.Tensor): first set of features in the cosine comparison
            second_features (torch.Tensor): second set of features in the cosine comparison

        Returns:
            (torch.Tensor): cosine similarity loss between the provided features.
        """
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the PerFCL loss, if it exists, using the configuration.

        Args:
            local_features (torch.Tensor): See PerFCL loss documentation
            old_local_features (torch.Tensor): See PerFCL loss documentation
            global_features (torch.Tensor): See PerFCL loss documentation
            old_global_features (torch.Tensor): See PerFCL loss documentation
            initial_global_features (torch.Tensor): See PerFCL loss documentation

        Returns:
            (tuple[torch.Tensor, torch.Tensor]): PerFCL loss based on the input values
        """
        assert self.perfcl_loss_config is not None
        global_feature_contrastive_loss, local_feature_contrastive_loss = self.perfcl_loss_config.perfcl_loss_function(
            local_features, old_local_features, global_features, old_global_features, initial_global_features
        )
        return (
            self.perfcl_loss_config.global_feature_contrastive_loss_weight * global_feature_contrastive_loss,
            self.perfcl_loss_config.local_feature_contrastive_loss_weight * local_feature_contrastive_loss,
        )
