from pathlib import Path
from typing import Optional, Sequence

import torch
from flwr.common.typing import Config

from fl4health.checkpointing.checkpointer import TorchCheckpointer
from fl4health.clients.basic_client import BasicClient
from fl4health.model_bases.fenda_base import FendaModel
from fl4health.parameter_exchange.layer_exchanger import FixedLayerExchanger
from fl4health.parameter_exchange.parameter_exchanger_base import ParameterExchanger
from fl4health.utils.losses import Losses, LossMeterType, SupConLoss
from fl4health.utils.metrics import Metric, MetricMeterType


class FendaClient(BasicClient):
    def __init__(
        self,
        data_path: Path,
        metrics: Sequence[Metric],
        device: torch.device,
        loss_meter_type: LossMeterType = LossMeterType.AVERAGE,
        metric_meter_type: MetricMeterType = MetricMeterType.AVERAGE,
        use_wandb_reporter: bool = False,
        checkpointer: Optional[TorchCheckpointer] = None,
    ) -> None:
        super().__init__(
            data_path, metrics, device, loss_meter_type, metric_meter_type, use_wandb_reporter, checkpointer
        )
        self.cos_sim = torch.nn.CosineSimilarity(dim=0)
        self.contrastive = SupConLoss()
        self.local_features: torch.Tensor
        self.global_features: torch.Tensor

    def get_parameter_exchanger(self, config: Config) -> ParameterExchanger:
        assert isinstance(self.model, FendaModel)
        return FixedLayerExchanger(self.model.layers_to_exchange())

    def predict(self, input: torch.Tensor) -> torch.Tensor:
        pred, self.local_features, self.global_features = self.model(input, self.pre_train)
        return pred

    def get_cosine_similarity_loss(self) -> torch.Tensor:

        assert len(self.local_features) == len(self.global_features)

        return torch.abs(self.cos_sim(self.local_features, self.global_features)).mean()

    def get_contrastive_loss(self) -> torch.Tensor:
        assert len(self.local_features) == len(self.global_features)
        labels = torch.cat((torch.ones(1), torch.zeros(1))).to(self.device)
        return self.contrastive(
            torch.cat((self.local_features.unsqueeze(0), self.global_features.unsqueeze(0)), dim=0), labels=labels
        )

    def compute_loss(self, preds: torch.Tensor, target: torch.Tensor) -> Losses:
        if self.pre_train:
            return super().compute_loss(preds, target)
        loss = self.criterion(preds, target)
        # cos_loss = self.get_cosine_similarity_loss()
        # total_loss = loss + 10 * cos_loss
        # losses = Losses(checkpoint=loss, backward=total_loss, additional_losses={"cos_sim_loss": cos_loss})
        contrastive_loss = self.get_contrastive_loss()
        total_loss = loss + 0.001 * contrastive_loss
        losses = Losses(checkpoint=loss, backward=total_loss, additional_losses={"contrastive_loss": contrastive_loss})
        return losses
