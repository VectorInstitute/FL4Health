import copy
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import torch
from flwr.common.typing import Config, NDArrays

from fl4health.checkpointing.checkpointer import TorchCheckpointer
from fl4health.clients.basic_client import BasicClient
from fl4health.model_bases.fenda_base import FendaModel
from fl4health.parameter_exchange.layer_exchanger import FixedLayerExchanger
from fl4health.parameter_exchange.parameter_exchanger_base import ParameterExchanger
from fl4health.utils.losses import Losses, LossMeterType
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
        self.perFCL_loss = False
        self.cos_sim_loss = False
        self.contrastive_loss = False
        self.cos_sim = torch.nn.CosineSimilarity(dim=-1)
        self.ce_criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.old_model: torch.nn.Module
        self.global_model: torch.nn.Module
        self.local_features: torch.Tensor
        self.shared_features: torch.Tensor
        self.local_old_features: torch.Tensor
        self.shared_old_features: torch.Tensor
        self.global_features: torch.Tensor
        self.temprature: float = 0.5

    def get_parameter_exchanger(self, config: Config) -> ParameterExchanger:
        assert isinstance(self.model, FendaModel)
        self.old_model = copy.deepcopy(self.model)
        self.old_model.eval()
        return FixedLayerExchanger(self.model.layers_to_exchange())

    def predict(self, input: torch.Tensor) -> Dict[str, torch.Tensor]:
        preds, self.local_features, self.shared_features = self.model(input)
        self.local_features = self.local_features.unsqueeze(2).unsqueeze(3)
        self.local_features = self.local_features.view(len(self.local_features), -1)
        self.shared_features = self.shared_features.view(len(self.shared_features), -1)
        _, self.local_old_features, self.shared_old_features = self.old_model(input)
        self.local_old_features = self.local_old_features.view(len(self.local_old_features), -1)
        self.shared_old_features = self.shared_old_features.view(len(self.shared_old_features), -1)
        if self.perFCL_loss:
            self.global_features = self.global_model.forward(input)
            self.global_features = self.global_features.view(len(self.global_features), -1)
        if isinstance(preds, dict):
            return preds
        elif isinstance(preds, torch.Tensor):
            return {"prediction": preds}
        else:
            raise ValueError("Model forward did not return a tensor or dictionary or tensors")

    def set_parameters(self, parameters: NDArrays, config: Config) -> None:
        output = super().set_parameters(parameters, config)
        if self.perFCL_loss:
            assert isinstance(self.model.global_module, torch.nn.Module)
            self.global_model = copy.deepcopy(self.model.global_module)
            self.global_model.eval()
        return output

    def get_cosine_similarity_loss(self) -> torch.Tensor:

        assert len(self.local_features) == len(self.shared_features)
        return torch.abs(self.cos_sim(self.local_features, self.shared_features)).mean()

    def get_contrastive_loss(self) -> torch.Tensor:
        assert len(self.local_features) == len(self.shared_features)
        posi = self.cos_sim(self.local_features, self.local_old_features)
        logits = posi.reshape(-1, 1)
        nega = self.cos_sim(self.local_features, self.shared_features)
        logits = torch.cat((logits, nega.reshape(-1, 1)), dim=1)
        logits /= self.temprature
        labels = torch.zeros(self.local_features.size(0)).to(self.device).long()

        return self.ce_criterion(logits, labels)

    def get_perFCL_loss(self) -> Tuple[torch.Tensor, torch.Tensor]:
        assert len(self.local_features) == len(self.global_features)

        posi = self.cos_sim(self.global_features, self.shared_features)
        logits_max = posi.reshape(-1, 1)
        nega = self.cos_sim(self.shared_features, self.shared_old_features)
        logits_max = torch.cat((logits_max, nega.reshape(-1, 1)), dim=1)
        logits_max /= self.temprature
        labels_max = torch.zeros(self.local_features.size(0)).to(self.device).long()

        posi = self.cos_sim(self.local_features, self.local_old_features)
        logits_min = posi.reshape(-1, 1)
        nega = self.cos_sim(self.local_features, self.global_features)
        logits_min = torch.cat((logits_min, nega.reshape(-1, 1)), dim=1)
        logits_min /= self.temprature
        labels_min = torch.zeros(self.local_features.size(0)).to(self.device).long()

        return self.ce_criterion(logits_min, labels_min), self.ce_criterion(logits_max, labels_max)

    def compute_loss(self, preds: Dict[str, torch.Tensor], target: torch.Tensor) -> Losses:
        if self.old_model is None:
            return super().compute_loss(preds, target)
        loss = self.criterion(preds, target)
        if self.cos_sim_loss:
            cos_loss = self.get_cosine_similarity_loss()
            total_loss = loss + 100 * cos_loss
            losses = Losses(checkpoint=loss, backward=total_loss, additional_losses={"cos_sim_loss": cos_loss})
        elif self.contrastive_loss:
            contrastive_loss = self.get_contrastive_loss()
            total_loss = loss + 10 * contrastive_loss
            losses = Losses(
                checkpoint=loss, backward=total_loss, additional_losses={"contrastive_loss": contrastive_loss}
            )
        elif self.perFCL_loss:
            contrastive_loss_minimize, contrastive_loss_maximize = self.get_perFCL_loss()
            total_loss = loss + 10 * contrastive_loss_minimize + 10 * contrastive_loss_maximize
            losses = Losses(
                checkpoint=loss,
                backward=total_loss,
                additional_losses={
                    "contrastive_loss_minimize": contrastive_loss_minimize,
                    "contrastive_loss_maximize": contrastive_loss_maximize,
                },
            )
        else:
            losses = Losses(
                checkpoint=loss,
                backward=loss,
            )

        return losses
