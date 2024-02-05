from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import torch
from flwr.common.typing import Config, NDArrays

from fl4health.checkpointing.checkpointer import TorchCheckpointer
from fl4health.clients.basic_client import BasicClient
from fl4health.losses.contrastive_loss import ContrastiveLoss
from fl4health.losses.mkmmd_loss import MkMmdLoss
from fl4health.model_bases.moon_base import MoonModel
from fl4health.utils.losses import Losses, LossMeterType
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
        self.temperature = temperature
        self.contrastive_loss = ContrastiveLoss(self.device, temperature=self.temperature)
        self.mkmmd_loss = MkMmdLoss(device=self.device, minimize_type_two_error=True).to(self.device)

        # Saving previous local models and global model at each communication round to compute contrastive loss
        self.len_old_models_buffer = len_old_models_buffer
        self.old_models_list: list[torch.nn.Module] = []
        self.global_model: torch.nn.Module

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
        old_features = torch.zeros(self.len_old_models_buffer, *features["features"].size()).to(self.device)
        for i, old_model in enumerate(self.old_models_list):
            _, old_model_features = old_model(input)
            old_features[i] = old_model_features["features"]
        _, global_model_features = self.global_model(input)
        features.update({"global_features": global_model_features["features"], "old_features": old_features})
        return preds, features

    def set_parameters(self, parameters: NDArrays, config: Config) -> None:
        assert isinstance(self.model, MoonModel)
        # Save the parameters of the old local model
        old_model = self.clone_and_freeze_model(self.model)
        self.old_models_list.append(old_model)
        if len(self.old_models_list) > self.len_old_models_buffer:
            self.old_models_list.pop(0)

        # Set the parameters of the model
        super().set_parameters(parameters, config)

        # Save the parameters of the global model
        self.global_model = self.clone_and_freeze_model(self.model)

        if self.mkmmd_loss_weights:
            self.set_optimized_betas(self.mkmmd_loss, self.old_models_list[-1], self.global_model)

    def set_optimized_betas(
        self, mkmmd_loss: MkMmdLoss, source_model: torch.nn.Module, target_model: torch.nn.Module
    ) -> None:
        """Set the optimized betas for the MK-MMD loss."""
        assert isinstance(source_model, MoonModel)
        assert isinstance(target_model, MoonModel)

        local_dist = []
        aggregated_dist = []

        # Compute the local and global features for the train loader
        self.model.eval()
        with torch.no_grad():
            for input, target in self.train_loader:
                input, target = input.to(self.device), target.to(self.device)
                _, source_features = source_model(input)
                _, target_features = target_model(input)
                local_dist.append(source_features["features"])
                aggregated_dist.append(target_features["features"])

        mkmmd_loss.betas = mkmmd_loss.optimize_betas(
            X=torch.cat(local_dist, dim=0), Y=torch.cat(aggregated_dist, dim=0), lambda_m=1e-5
        )

    def compute_loss(
        self, preds: Dict[str, torch.Tensor], features: Dict[str, torch.Tensor], target: torch.Tensor
    ) -> Losses:
        """
        Computes loss given predictions and features of the model and ground truth data. Loss includes
        base loss plus a model contrastive loss.

        Args:
            preds (Dict[str, torch.Tensor]): Prediction(s) of the model(s) indexed by name.
                All predictions included in dictionary will be used to compute metrics.
            features: (Dict[str, torch.Tensor]): Feature(s) of the model(s) indexed by name.
            target: (torch.Tensor): Ground truth data to evaluate predictions against.

        Returns:
            Losses: Object containing checkpoint loss, backward loss and additional losses indexed by name.
        """
        if len(self.old_models_list) == 0:
            return super().compute_loss(preds, features, target)

        loss = self.criterion(preds["prediction"], target)
        total_loss = loss.clone()
        additional_losses = {}

        if self.contrastive_weight:
            contrastive_loss = self.contrastive_loss(
                features["features"], features["global_features"].unsqueeze(0), features["old_features"]
            )
            total_loss += self.contrastive_weight * contrastive_loss
            additional_losses["contrastive_loss"] = contrastive_loss
        elif self.mkmmd_loss_weights:
            min_mkmmd_loss = self.mkmmd_loss(features["features"], features["global_features"])
            max_mkmmd_loss = self.mkmmd_loss(features["features"], features["old_features"][-1])
            total_loss += self.mkmmd_loss_weights[0] * min_mkmmd_loss - self.mkmmd_loss_weights[1] * max_mkmmd_loss
            additional_losses["min_mkmmd_loss"] = min_mkmmd_loss
            additional_losses["max_mkmmd_loss"] = max_mkmmd_loss
        losses = Losses(checkpoint=loss, backward=total_loss, additional_losses=additional_losses)
        return losses
