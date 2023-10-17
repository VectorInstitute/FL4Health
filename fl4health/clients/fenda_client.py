import copy
from pathlib import Path
from typing import Optional, Sequence, Tuple

import torch
from flwr.common.typing import Config, NDArrays

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
        self.perFCL_loss = True
        self.cos_sim_loss = False
        self.contrastive_loss = False
        self.cos_sim = torch.nn.CosineSimilarity(dim=0)
        self.contrastive = SupConLoss()
        self.local_features: torch.Tensor
        self.global_features: torch.Tensor
        self.global_old_features: torch.Tensor
        self.global_model: torch.nn.Module

    def get_parameter_exchanger(self, config: Config) -> ParameterExchanger:
        assert isinstance(self.model, FendaModel)
        return FixedLayerExchanger(self.model.layers_to_exchange())

    def predict(self, input: torch.Tensor) -> torch.Tensor:
        pred, self.local_features, self.global_features = self.model(input, self.pre_train)
        if self.perFCL_loss:
            self.global_old_features = self.global_model.forward(input)
        return pred

    def set_parameters(self, parameters: NDArrays, config: Config) -> None:
        output = super().set_parameters(parameters, config)
        assert isinstance(self.model.global_module, torch.nn.Module)
        self.global_model = copy.deepcopy(self.model.global_module)
        self.global_model.eval()
        return output

    def get_cosine_similarity_loss(self) -> torch.Tensor:

        assert len(self.local_features) == len(self.global_features)

        return torch.abs(self.cos_sim(self.local_features, self.global_features)).mean()

    def get_contrastive_loss(self) -> torch.Tensor:
        assert len(self.local_features) == len(self.global_features)
        labels = torch.cat((torch.ones(1), torch.zeros(1))).to(self.device)
        return self.contrastive(
            torch.cat((self.local_features.unsqueeze(0), self.global_features.unsqueeze(0)), dim=0), labels=labels
        )

    def get_perFCL_loss(self) -> Tuple[torch.Tensor, torch.Tensor]:
        assert len(self.local_features) == len(self.global_features)
        labels_minimize = torch.cat((torch.ones(1), torch.zeros(1))).to(self.device)
        labels_maximize = torch.arange(len(self.local_features)).to(self.device)
        return self.contrastive(
            torch.cat((self.local_features.unsqueeze(0), self.global_old_features.unsqueeze(0)), dim=0),
            labels=labels_minimize,
        ), self.contrastive(
            torch.cat((self.global_features.unsqueeze(1), self.global_old_features.unsqueeze(1)), dim=1),
            labels=labels_maximize,
        )

    def compute_loss(self, preds: torch.Tensor, target: torch.Tensor) -> Losses:
        if self.pre_train:
            return super().compute_loss(preds, target)
        loss = self.criterion(preds, target)
        if self.cos_sim_loss:
            cos_loss = self.get_cosine_similarity_loss()
            total_loss = loss + 10 * cos_loss
            losses = Losses(checkpoint=loss, backward=total_loss, additional_losses={"cos_sim_loss": cos_loss})
        if self.contrastive_loss:
            contrastive_loss = self.get_contrastive_loss()
            total_loss = loss + 0.001 * contrastive_loss
            losses = Losses(
                checkpoint=loss, backward=total_loss, additional_losses={"contrastive_loss": contrastive_loss}
            )
        if self.perFCL_loss:
            contrastive_loss_minimize, contrastive_loss_maximize = self.get_perFCL_loss()
            total_loss = loss + 0.001 * contrastive_loss_minimize + 0.001 * contrastive_loss_maximize
            losses = Losses(
                checkpoint=loss,
                backward=total_loss,
                additional_losses={
                    "contrastive_loss_minimize": contrastive_loss_minimize,
                    "contrastive_loss_maximize": contrastive_loss_maximize,
                },
            )

        return losses
