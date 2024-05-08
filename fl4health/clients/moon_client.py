from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import torch

from fl4health.checkpointing.client_module import ClientCheckpointModule
from fl4health.clients.basic_client import BasicClient, TorchInputType
from fl4health.model_bases.sequential_split_models import SequentiallySplitModel
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
        checkpointer: Optional[ClientCheckpointModule] = None,
        temperature: float = 0.5,
        contrastive_weight: Optional[float] = None,
        len_old_models_buffer: int = 1,
    ) -> None:
        super().__init__(
            data_path=data_path,
            metrics=metrics,
            device=device,
            loss_meter_type=loss_meter_type,
            checkpointer=checkpointer,
        )
        self.cos_sim = torch.nn.CosineSimilarity(dim=-1).to(self.device)
        self.ce_criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.contrastive_weight = contrastive_weight
        self.temperature = temperature

        # Saving previous local models and global model at each communication round to compute contrastive loss
        self.len_old_models_buffer = len_old_models_buffer
        self.old_models_list: list[torch.nn.Module] = []
        self.global_model: Optional[torch.nn.Module] = None

    def predict(self, input: TorchInputType) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Computes the prediction(s) and features of the model(s) given the input.

        Args:
            input (TorchInputType): Inputs to be fed into the model. TorchInputType is simply an alias
            for the union of torch.Tensor and Dict[str, torch.Tensor].

        Returns:
            Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]: A tuple in which the first element
            contains predictions indexed by name and the second element contains intermediate activations
            index by name. Specifically the features of the model, features of the global model and features of
            the old model are returned. All predictions included in dictionary will be used to compute metrics.
        """
        assert isinstance(input, torch.Tensor)
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

    def get_contrastive_loss(
        self, features: torch.Tensor, global_features: torch.Tensor, old_features: torch.Tensor
    ) -> torch.Tensor:
        """
        This contrastive loss is implemented based on https://github.com/QinbinLi/MOON.
        The primary idea is to enhance the similarity between the current local features and the global feature
        as positive pairs while reducing the similarity between the current local features and the previous local
        features as negative pairs.
        """
        assert len(features) == len(global_features)
        posi = self.cos_sim(features, global_features)
        logits = posi.reshape(-1, 1)
        for old_feature in old_features:
            assert len(features) == len(old_feature)
            nega = self.cos_sim(features, old_feature)
            logits = torch.cat((logits, nega.reshape(-1, 1)), dim=1)
        logits /= self.temperature
        labels = torch.zeros(features.size(0)).to(self.device).long()

        return self.ce_criterion(logits, labels)

    def update_after_train(self, local_steps: int, loss_dict: Dict[str, float]) -> None:
        assert isinstance(self.model, SequentiallySplitModel)
        # Save the parameters of the old LOCAL model
        old_model = self.clone_and_freeze_model(self.model)
        self.old_models_list.append(old_model)
        if len(self.old_models_list) > self.len_old_models_buffer:
            self.old_models_list.pop(0)

        return super().update_after_train(local_steps, loss_dict)

    def update_before_train(self, current_server_round: int) -> None:
        # Save the parameters of the global model
        self.global_model = self.clone_and_freeze_model(self.model)

        return super().update_before_train(current_server_round)

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

        if self.contrastive_weight:
            contrastive_loss = self.get_contrastive_loss(
                features["features"], features["global_features"], features["old_features"]
            )
            total_loss += self.contrastive_weight * contrastive_loss
            additional_losses["contrastive_loss"] = contrastive_loss

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
