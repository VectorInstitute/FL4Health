import copy
from pathlib import Path
from typing import Dict, Optional, Sequence

import torch
from flwr.common.typing import Config, NDArrays

from fl4health.checkpointing.checkpointer import TorchCheckpointer
from fl4health.clients.basic_client import BasicClient
from fl4health.model_bases.moon_base import MoonModel
from fl4health.utils.losses import Losses, LossMeterType
from fl4health.utils.metrics import Metric, MetricMeterType


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
        metric_meter_type: MetricMeterType = MetricMeterType.AVERAGE,
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
            metric_meter_type=metric_meter_type,
            checkpointer=checkpointer,
        )

        self.contrastive_weight = contrastive_weight
        self.temperature = temperature
        self.len_old_models_buffer = len_old_models_buffer
        self.cos_sim = torch.nn.CosineSimilarity(dim=-1)
        self.ce_criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.old_models_list: list[MoonModel] = []
        self.global_model: MoonModel

    def predict(self, input: torch.Tensor) -> Dict[str, torch.Tensor]:
        preds = self.model(input)
        preds["old_features"] = torch.zeros(self.len_old_models_buffer, *preds["features"].size()).to(self.device)
        for i, old_model in enumerate(self.old_models_list):
            old_preds = old_model(input)
            preds["old_features"][i] = old_preds["features"]
        global_preds = self.global_model(input)
        preds["global_features"] = global_preds["features"]
        if isinstance(preds, dict):
            return preds
        elif isinstance(preds, torch.Tensor):
            return {"prediction": preds}
        else:
            raise ValueError("Model forward did not return a tensor or dictionary or tensors")

    def get_contrastive_loss(
        self, features: torch.Tensor, global_features: torch.Tensor, old_features: torch.Tensor
    ) -> torch.Tensor:
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
        old_model = copy.deepcopy(self.model)
        for param in old_model.parameters():
            param.requires_grad = False
        old_model.eval()
        self.old_models_list.append(old_model)
        if len(self.old_models_list) > self.len_old_models_buffer:
            self.old_models_list.pop(0)

        # Set the parameters of the model
        output = super().set_parameters(parameters, config)

        # Save the parameters of the global model
        self.global_model = copy.deepcopy(self.model)
        for param in self.global_model.parameters():
            param.requires_grad = False
        self.global_model.eval()
        return output

    def compute_loss(self, preds: Dict[str, torch.Tensor], target: torch.Tensor) -> Losses:
        if len(self.old_models_list) == 0:
            return super().compute_loss(preds, target)
        loss = self.criterion(preds["prediction"], target)
        contrastive_loss = self.get_contrastive_loss(
            preds["features"], preds["global_features"], preds["old_features"]
        )
        total_loss = loss + self.contrastive_weight * contrastive_loss
        losses = Losses(checkpoint=loss, backward=total_loss, additional_losses={"contrastive_loss": contrastive_loss})
        return losses
