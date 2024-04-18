from logging import INFO, ERROR
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import torch
from flwr.common.logger import log

from fl4health.checkpointing.checkpointer import TorchCheckpointer
from fl4health.clients.moon_client import MoonClient
from fl4health.losses.mkmmd_loss import MkMmdLoss
from fl4health.model_bases.moon_base import MoonModel
from fl4health.utils.losses import  LossMeterType
from fl4health.utils.metrics import Metric


class MoonMkmmdClient(MoonClient):
    """
    This client implements the MOON algorithm from Model-Contrastive Federated Learning. The key idea of MOON
    is to utilize the similarity between model representations to correct the local training of individual parties,
    i.e., conducting contrastive learning in model-level.
    """

    def __init__(
        self,
        data_path: Path,
        metrics: Sequence[Metric],
        device: torch.device,
        loss_meter_type: LossMeterType = LossMeterType.AVERAGE,
        temperature: float = 0.5,
        len_old_models_buffer: int = 1,
        beta_update_interval: int = 20,
        checkpointer: Optional[TorchCheckpointer] = None,
        contrastive_weight: Optional[float] = None,
        mkmmd_loss_weights: Optional[Tuple[float, float]] = None,
        feature_l2_norm_weight: Optional[float] = None,
    ) -> None:
        super().__init__(
            data_path=data_path,
            metrics=metrics,
            device=device,
            loss_meter_type=loss_meter_type,
            checkpointer=checkpointer,
            contrastive_weight=contrastive_weight,
            temperature=temperature,
            len_old_models_buffer=len_old_models_buffer,
        )
        self.mkmmd_loss_weights = mkmmd_loss_weights
        if self.mkmmd_loss_weights == (0,0):
            log(
                ERROR,
                "MK-MMD loss weight is set to (0,0). As none of MK-MMD losses will not be computed, ",
                "please use vanilla MoonClient instead.",
            )
        self.feature_l2_norm_weight = feature_l2_norm_weight
        self.mkmmd_loss_min = MkMmdLoss(device=self.device, minimize_type_two_error=True).to(self.device)
        self.mkmmd_loss_max = MkMmdLoss(device=self.device, minimize_type_two_error=False).to(self.device)
        self.beta_update_interval = beta_update_interval


        self.betas_optimized = False


    def update_before_train(self, current_server_round: int) -> None:

        # Optimizing betas for the MK-MMD loss has not been done yet
        self.betas_optimized = False

        return super().update_before_train(current_server_round)

    def update_after_step(self, step: int) -> None:
        if step > 0 and step % self.beta_update_interval == 0:
            if self.mkmmd_loss_weights and len(self.old_models_list) > 0 and self.global_model:
                # Get the feature distribution of the old, local and global models with evaluation mode
                local_distribution, global_distribution, old_distribution = self.update_buffers(
                    self.model, self.global_model, self.old_models_list[-1]
                )
                # Update betas for the MK-MMD loss
                if self.mkmmd_loss_weights[0] != 0:
                    self.mkmmd_loss_min.betas = self.mkmmd_loss_min.optimize_betas(
                        X=local_distribution, Y=global_distribution, lambda_m=1e-5
                    )
                    log(INFO, f"Set optimized betas to minimize distance: {self.mkmmd_loss_min.betas.squeeze()}.")

                if self.mkmmd_loss_weights[1] != 0:
                    self.mkmmd_loss_max.betas = self.mkmmd_loss_max.optimize_betas(
                        X=local_distribution,
                        Y=old_distribution,
                        lambda_m=1e-5,
                    )
                    log(INFO, f"Set optimized betas to maximize distance: {self.mkmmd_loss_max.betas.squeeze()}.")

                # Betas have been optimized
                self.betas_optimized = True

        return super().update_after_step(step)

    def update_buffers(
        self, local_model: torch.nn.Module, global_model: torch.nn.Module, old_local_model: torch.nn.Module
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Update the feature buffer of the local and global features."""
        assert isinstance(local_model, MoonModel)
        assert isinstance(global_model, MoonModel)
        assert isinstance(old_local_model, MoonModel)

        local_buffer = []
        global_buffer = []
        old_buffer = []

        init_state_local_model = local_model.training

        # Set the models to evaluation mode
        local_model.eval()
        global_model.eval()
        old_local_model.eval()
        assert not local_model.training
        assert not global_model.training
        assert not old_local_model.training

        with torch.no_grad():
            for input, target in self.train_loader:
                input, target = input.to(self.device), target.to(self.device)
                _, local_features = local_model(input)
                _, global_features = global_model(input)
                _, old_local_features = old_local_model(input)

                local_buffer.append(local_features["features"].reshape(len(local_features["features"]), -1))
                global_buffer.append(global_features["features"].reshape(len(global_features["features"]), -1))
                old_buffer.append(old_local_features["features"].reshape(len(old_local_features["features"]), -1))

        # Set the local model back to their original mode
        if init_state_local_model:
            local_model.train()

        return torch.cat(local_buffer, dim=0), torch.cat(global_buffer, dim=0), torch.cat(old_buffer, dim=0)

    def compute_loss_and_additional_losses(
        self,
        preds: Dict[str, torch.Tensor],
        features: Dict[str, torch.Tensor],
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Computes the loss and any additional losses given predictions of the model and ground truth data.
        For MOON, the loss is the total loss and the additional losses are the loss, contrastive loss, and total loss.

        Args:
            preds (Dict[str, torch.Tensor]): Prediction(s) of the model(s) indexed by name.
            features (Dict[str, torch.Tensor]): Feature(s) of the model(s) indexed by name.
            target (torch.Tensor): Ground truth data to evaluate predictions against.

        Returns:
            Tuple[torch.Tensor, Union[Dict[str, torch.Tensor], None]]; A tuple with:
                - The tensor for the total loss
                - A dictionary with `loss`, `contrastive_loss` and `total loss` keys and their calculated values.
        """

        total_loss, additional_losses = super().compute_loss_and_additional_losses(preds, features, target)

        if (
            self.mkmmd_loss_weights
            and "global_features" in features
            and "old_features" in features
            and self.betas_optimized
        ):
            if self.mkmmd_loss_weights[0] != 0:
                mkmmd_loss_min = self.mkmmd_loss_min(features["features"], features["global_features"])
                total_loss += self.mkmmd_loss_weights[0] * mkmmd_loss_min
                additional_losses["mkmmd_loss_min"] = mkmmd_loss_min

            if self.mkmmd_loss_weights[1] != 0:
                mkmmd_loss_max = self.mkmmd_loss_max(features["features"], features["old_features"][-1])
                total_loss -= self.mkmmd_loss_weights[1] * mkmmd_loss_max
                additional_losses["mkmmd_loss_max"] = mkmmd_loss_max

            if self.feature_l2_norm_weight:
                # Compute the average L2 norm of the features over the batch
                feature_l2_norm_loss = torch.linalg.norm(features["features"]) / len(features["features"])
                total_loss += self.feature_l2_norm_weight * feature_l2_norm_loss
                additional_losses["feature_l2_norm_loss"] = feature_l2_norm_loss

        additional_losses["total_loss"] = total_loss

        return total_loss, additional_losses
