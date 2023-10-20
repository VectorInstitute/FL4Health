import copy
from pathlib import Path
from typing import List, Optional, Sequence

import torch
from flwr.common.typing import Config, NDArrays

from fl4health.checkpointing.checkpointer import TorchCheckpointer
from fl4health.clients.basic_client import BasicClient
from fl4health.model_bases.moon_base import MoonModel
from fl4health.parameter_exchange.parameter_exchanger_base import ParameterExchanger
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
    ) -> None:
        super().__init__(
            data_path=data_path,
            metrics=metrics,
            device=device,
            loss_meter_type=loss_meter_type,
            metric_meter_type=metric_meter_type,
            checkpointer=checkpointer,
        )
        self.initial_tensors: List[torch.Tensor]
        self.contrastive_weight: float = 10
        self.current_loss: float
        self.cos_sim = torch.nn.CosineSimilarity(dim=-1)
        self.ce_criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.old_models_list: list[MoonModel] = []
        self.global_model: MoonModel
        self.len_old_models_buffer: int = 1
        self.temprature: float = 0.5
        self.features: torch.Tensor
        self.old_features_list: list[torch.Tensor]
        self.global_features: torch.Tensor

    def predict(self, input: torch.Tensor) -> torch.Tensor:
        pred, self.features, _ = self.model(input)
        self.features = self.features.view(len(self.features), -1)
        for old_model in self.old_models_list:
            _, old_features, _ = old_model(input)
            old_features = old_features.view(len(old_features), -1)
            self.old_features_list.append(old_features)
        _, global_features, _ = self.global_model(input)
        global_features = global_features.view(len(global_features), -1)
        return pred

    def get_contrastive_loss(self) -> torch.Tensor:
        assert len(self.features) == len(self.global_features)
        posi = self.cos_sim(self.features, self.global_features)
        logits = posi.reshape(-1, 1)
        for old_features in self.old_features_list:
            assert len(self.features) == len(old_features)
            nega = self.cos_sim(self.features, old_features)
            logits = torch.cat((logits, nega.reshape(-1, 1)), dim=1)
        logits /= self.temprature
        labels = torch.zeros(self.features.size(0)).to(self.device).long()

        return self.ce_criterion(logits, labels)

    def get_parameter_exchanger(self, config: Config) -> ParameterExchanger:
        assert isinstance(self.model, MoonModel)
        old_model = copy.deepcopy(self.model)
        old_model.eval()
        self.old_models_list.append(old_model)
        if len(self.old_models_list) > self.len_old_models_buffer:
            self.old_models_list.pop(0)
        return super().get_parameter_exchanger(config)

    def set_parameters(self, parameters: NDArrays, config: Config) -> None:
        output = super().set_parameters(parameters, config)
        assert isinstance(self.model, MoonModel)
        self.global_model = copy.deepcopy(self.model)
        self.global_model.eval()
        return output

    def compute_loss(self, preds: torch.Tensor, target: torch.Tensor) -> Losses:
        if len(self.old_models_list) == 0:
            return super().compute_loss(preds, target)
        loss = self.criterion(preds, target)
        contrastive_loss = self.get_contrastive_loss()
        total_loss = loss + self.contrastive_weight * contrastive_loss
        losses = Losses(checkpoint=loss, backward=total_loss, additional_losses={"contrastive_loss": contrastive_loss})
        return losses
