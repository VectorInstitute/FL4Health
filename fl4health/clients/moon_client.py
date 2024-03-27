from logging import INFO
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import torch
from flwr.common.logger import log

from fl4health.checkpointing.checkpointer import TorchCheckpointer
from fl4health.clients.basic_client import BasicClient
from fl4health.losses.contrastive_loss import ContrastiveLoss
from fl4health.losses.mkmmd_loss import MkMmdLoss
from fl4health.model_bases.moon_base import MoonModel
from fl4health.utils.losses import EvaluationLosses, LossMeterType, TrainingLosses
from fl4health.utils.metrics import Metric


class MoonClient(BasicClient):
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
        feature_l2_norm: Optional[float] = None,
    ) -> None:
        super().__init__(
            data_path=data_path,
            metrics=metrics,
            device=device,
            loss_meter_type=loss_meter_type,
            checkpointer=checkpointer,
        )
        self.contrastive_weight = contrastive_weight
        self.mkmmd_loss_weights = mkmmd_loss_weights
        self.feature_l2_norm = feature_l2_norm
        self.contrastive_loss = ContrastiveLoss(self.device, temperature=temperature)
        self.mkmmd_loss_min = MkMmdLoss(device=self.device, minimize_type_two_error=True).to(self.device)
        self.mkmmd_loss_max = MkMmdLoss(device=self.device, minimize_type_two_error=False).to(self.device)
        self.beta_update_interval = beta_update_interval

        # Saving previous local models and global model at each communication round to compute contrastive loss
        self.len_old_models_buffer = len_old_models_buffer
        self.old_models_list: list[torch.nn.Module] = []
        self.global_model: Optional[torch.nn.Module] = None
        self.betas_optimized = False

    def predict(self, input: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Computes the prediction(s) and features of the model(s) given the input.

        Args:
            input (torch.Tensor): Inputs to be fed into the model.

        Returns:
            Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]: A tuple in which the first element
            contains predictions indexed by name and the second element contains intermediate activations
            index by name. Specifically the features of the model, features of the global model and features of
            the old model are returned. All predictions included in dictionary will be used to compute metrics.
        """
        preds, features = self.model(input)
        # If there are no models in the old_models_list, we don't compute the features for the contrastive loss
        if len(self.old_models_list) > 0:
            old_features = torch.zeros(len(self.old_models_list), *features["features"].size()).to(self.device)
            for i, old_model in enumerate(self.old_models_list):
                _, old_model_features = old_model(input)
                old_features[i] = old_model_features["features"]
            features.update({"old_features": old_features})
        if self.global_model is not None:
            _, global_model_features = self.global_model(input)
            features.update({"global_features": global_model_features["features"]})

        return preds, features

    def update_after_train(self, local_steps: int, loss_dict: Dict[str, float]) -> None:
        assert isinstance(self.model, MoonModel)
        # Save the parameters of the old LOCAL model
        old_model = self.clone_and_freeze_model(self.model)
        self.old_models_list.append(old_model)
        if len(self.old_models_list) > self.len_old_models_buffer:
            self.old_models_list.pop(0)

        return super().update_after_train(local_steps, loss_dict)

    def update_before_train(self, current_server_round: int) -> None:
        # Save the parameters of the global model
        self.global_model = self.clone_and_freeze_model(self.model)

        # Optimizing betas for the MK-MMD loss has not been done yet
        self.betas_optimized = False

        return super().update_before_train(current_server_round)

    def update_after_step(self, step: int) -> None:
        if step > 0 and step % self.beta_update_interval == 0:
            if self.mkmmd_loss_weights and len(self.old_models_list) > 0 and self.global_model:
                # Get the feature distribution of the old, local and global features with evaluation mode
                local_distribution, global_distribution, old_distribution = self.update_buffers(
                    self.model, self.global_model, self.old_models_list[-1]
                )
                # Update betas for the MK-MMD loss based on gathered features during training
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

        # Compute the old features before aggregation and global features
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

        loss = self.criterion(preds["prediction"], target)
        total_loss = loss.clone()
        additional_losses = {
            "loss": loss,
        }

        if self.contrastive_weight and "old_features" in features:
            contrastive_loss = self.contrastive_loss(
                features["features"], features["global_features"].unsqueeze(0), features["old_features"]
            )
            total_loss += self.contrastive_weight * contrastive_loss
            additional_losses["contrastive_loss"] = contrastive_loss

        elif (
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

            if self.feature_l2_norm:
                feature_l2_norm_loss = torch.linalg.norm(features["features"], p=2)
                total_loss += self.feature_l2_norm * feature_l2_norm_loss
                additional_losses["feature_l2_norm_loss"] = feature_l2_norm_loss

        additional_losses["total_loss"] = total_loss

        return total_loss, additional_losses

    def compute_training_loss(
        self,
        preds: Dict[str, torch.Tensor],
        features: Dict[str, torch.Tensor],
        target: torch.Tensor,
    ) -> TrainingLosses:
        """
        Computes training loss given predictions and features of the model and ground truth data. Loss includes
        base loss plus a model contrastive loss.

        Args:
            preds (Dict[str, torch.Tensor]): Prediction(s) of the model(s) indexed by name.
                All predictions included in dictionary will be used to compute metrics.
            features: (Dict[str, torch.Tensor]): Feature(s) of the model(s) indexed by name.
            target: (torch.Tensor): Ground truth data to evaluate predictions against.

        Returns:
            TrainingLosses: an instance of TrainingLosses containing backward loss and additional losses
            indexed by name.
        """

        # Check that the model is in training mode
        assert self.model.training

        # If there are no old local models in the list (first pass of MOON training), we just do basic loss
        #  calculations
        if len(self.old_models_list) == 0:
            total_loss, additional_losses = super().compute_loss_and_additional_losses(preds, features, target)
        else:
            total_loss, additional_losses = self.compute_loss_and_additional_losses(preds, features, target)
        return TrainingLosses(backward=total_loss, additional_losses=additional_losses)

    def compute_evaluation_loss(
        self,
        preds: Dict[str, torch.Tensor],
        features: Dict[str, torch.Tensor],
        target: torch.Tensor,
    ) -> EvaluationLosses:
        """
        Computes evaluation loss given predictions and features of the model and ground truth data. Loss includes
        base loss plus a model contrastive loss.

        Args:
            preds (Dict[str, torch.Tensor]): Prediction(s) of the model(s) indexed by name.
                All predictions included in dictionary will be used to compute metrics.
            features: (Dict[str, torch.Tensor]): Feature(s) of the model(s) indexed by name.
            target: (torch.Tensor): Ground truth data to evaluate predictions against.

        Returns:
            EvaluationLosses: an instance of EvaluationLosses containing checkpoint loss and
                additional losses indexed by name.
        """
        # Check that the model is in evaluation mode
        assert not self.model.training

        # If there are no old local models in the list (first pass of MOON training), we just do basic loss
        # calculations
        if len(self.old_models_list) == 0:
            checkpoint_loss, additional_losses = super().compute_loss_and_additional_losses(preds, features, target)
        else:
            _, additional_losses = self.compute_loss_and_additional_losses(preds, features, target)
            # Note that the first parameter returned is the "total loss", which includes the contrastive loss
            # So we use the vanilla loss stored in additional losses for checkpointing.
            checkpoint_loss = additional_losses["loss"]
        return EvaluationLosses(checkpoint=checkpoint_loss, additional_losses=additional_losses)
