from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import torch
from flwr.common.typing import Config, NDArrays

from fl4health.checkpointing.checkpointer import TorchCheckpointer
from fl4health.clients.basic_client import BasicClient
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
        checkpointer: Optional[TorchCheckpointer] = None,
        temperature: float = 0.5,
        contrastive_weight: float = 10,
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

    def get_contrastive_loss(
        self, features: torch.Tensor, global_features: torch.Tensor, old_features: torch.Tensor
    ) -> torch.Tensor:
        """
        This constrastive loss is implemented based on https://github.com/QinbinLi/MOON.
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
        contrastive_loss = self.get_contrastive_loss(
            features["features"], features["global_features"], features["old_features"]
        )
        total_loss = loss + self.contrastive_weight * contrastive_loss
        losses = Losses(checkpoint=loss, backward=total_loss, additional_losses={"contrastive_loss": contrastive_loss})
        return losses
