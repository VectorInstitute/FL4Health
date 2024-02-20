from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import torch

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
        checkpointer: Optional[TorchCheckpointer] = None,
        contrastive_weight: Optional[float] = None,
        mkmmd_loss_weights: Optional[Tuple[float, float]] = None,
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
        self.contrastive_loss = ContrastiveLoss(self.device, temperature=temperature)
        self.mkmmd_loss = MkMmdLoss(device=self.device, minimize_type_two_error=True).to(self.device)

        # Saving previous local models and global model at each communication round to compute contrastive loss
        self.len_old_models_buffer = len_old_models_buffer
        self.old_models_list: list[torch.nn.Module] = []
        self.global_model: Optional[torch.nn.Module] = None

    def predict(self, input: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Computes the prediction(s) and features of the model(s) given the input.

        Args:
            input (torch.Tensor): Inputs to be fed into the model.

        Returns:
            Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]: A tuple in which the first element
            contains predictions indexed by name and the second element contains intermediate activations
            index by name. Specificaly the features of the model, features of the global model and features of
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

        if self.mkmmd_loss_weights and len(self.old_models_list) > 0:
            self.set_optimized_betas(self.mkmmd_loss, self.old_models_list[-1], self.global_model)

        return super().update_before_train(current_server_round)

    def set_optimized_betas(
        self, mkmmd_loss: MkMmdLoss, old_model: torch.nn.Module, global_model: torch.nn.Module
    ) -> None:
        """Set the optimized betas for the MK-MMD loss."""
        assert isinstance(old_model, MoonModel)
        assert isinstance(global_model, MoonModel)

        old_distribution = []
        global_distribution = []

        # Compute the old features before aggregation and global features
        old_model.eval()
        global_model.eval()
        with torch.no_grad():
            for input, target in self.train_loader:
                input, target = input.to(self.device), target.to(self.device)
                _, old_features = old_model(input)
                _, global_features = global_model(input)
                old_distribution.append(old_features["features"])
                global_distribution.append(global_features["features"])

        mkmmd_loss.betas = mkmmd_loss.optimize_betas(
            X=torch.cat(old_distribution, dim=0), Y=torch.cat(global_distribution, dim=0), lambda_m=1e-5
        )

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

        elif self.mkmmd_loss_weights and "old_features" in features:
            min_mkmmd_loss = self.mkmmd_loss(features["features"], features["global_features"])
            max_mkmmd_loss = self.mkmmd_loss(features["features"], features["old_features"][-1])
            total_loss += (
                self.mkmmd_loss_weights[0] * min_mkmmd_loss.sum() - self.mkmmd_loss_weights[1] * max_mkmmd_loss.sum()
            )
            additional_losses["min_mkmmd_loss"] = min_mkmmd_loss
            additional_losses["max_mkmmd_loss"] = max_mkmmd_loss

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
