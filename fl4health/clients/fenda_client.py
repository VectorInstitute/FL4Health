import copy
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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
        temperature: float = 0.5,
        perFCL_loss_weights: List[float] = [0.0, 0.0],
        cos_sim_loss_weight: float = 0.0,
        contrastive_loss_weight: float = 0.0,
    ) -> None:
        super().__init__(
            data_path=data_path,
            metrics=metrics,
            device=device,
            loss_meter_type=loss_meter_type,
            metric_meter_type=metric_meter_type,
            checkpointer=checkpointer,
        )
        self.perFCL_loss_weights = perFCL_loss_weights
        self.cos_sim_loss_weight = cos_sim_loss_weight
        self.contrastive_loss_weight = contrastive_loss_weight
        self.cos_sim = torch.nn.CosineSimilarity(dim=-1)
        self.ce_criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.temperature: float = temperature

        # Need to save previous local module, global module and aggregated global module at each communication round
        # to compute contrastive loss.
        self.old_local_module: torch.nn.Module
        self.old_global_module: torch.nn.Module
        self.aggregated_global_module: torch.nn.Module

    def get_parameter_exchanger(self, config: Config) -> ParameterExchanger:
        assert isinstance(self.model, FendaModel)
        return FixedLayerExchanger(self.model.layers_to_exchange())

    def predict(self, input: torch.Tensor) -> Dict[str, torch.Tensor]:

        preds = self.model(input)

        preds["old_local_features"] = self.old_local_module.forward(input).view(len(preds["local_features"]), -1)
        preds["old_global_features"] = self.old_global_module.forward(input).view(len(preds["global_features"]), -1)
        if self.perFCL_loss_weights[0] != 0.0 or self.perFCL_loss_weights[1] != 0.0:
            preds["aggregated_global_features"] = self.aggregated_global_module.forward(input).view(
                len(preds["global_features"]), -1
            )

        if isinstance(preds, dict):
            return preds
        elif isinstance(preds, torch.Tensor):
            return {"prediction": preds}
        else:
            raise ValueError("Model forward did not return a tensor or dictionary or tensors")

    def set_parameters(self, parameters: NDArrays, config: Config) -> None:

        # Save the parameters of the old model
        assert isinstance(self.model.local_module, torch.nn.Module)
        assert isinstance(self.model.global_module, torch.nn.Module)
        self.old_local_module = copy.deepcopy(self.model.local_module)
        self.old_global_module = copy.deepcopy(self.model.global_module)
        for param in self.old_local_module.parameters():
            param.requires_grad = False
        for param in self.old_global_module.parameters():
            param.requires_grad = False
        self.old_local_module.eval()
        self.old_global_module.eval()

        # Set the parameters of the model
        output = super().set_parameters(parameters, config)

        # Save the parameters of the global model
        if self.perFCL_loss_weights[0] != 0.0 or self.perFCL_loss_weights[1] != 0.0:
            assert isinstance(self.model.global_module, torch.nn.Module)
            self.aggregated_global_module = copy.deepcopy(self.model.global_module)
            for param in self.aggregated_global_module.parameters():
                param.requires_grad = False
            self.aggregated_global_module.eval()
        return output

    def get_cosine_similarity_loss(self, local_features: torch.Tensor, global_features: torch.Tensor) -> torch.Tensor:
        """
        Cosine similarity loss aims to minimize the similarity among current local features and current global
        features of fenda model.
        """
        assert len(local_features) == len(global_features)
        return torch.abs(self.cos_sim(local_features, global_features)).mean()

    def get_contrastive_loss(
        self, local_features: torch.Tensor, old_local_features: torch.Tensor, global_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Contrastive loss airms to enhance the similarity between the current local features and old local feature
        as positive pairs while reducing the similarity between the current local features and current global
        features as negative pairs.
        """
        assert len(local_features) == len(old_local_features)
        posi = self.cos_sim(local_features, old_local_features)
        logits = posi.reshape(-1, 1)
        assert len(local_features) == len(global_features)
        nega = self.cos_sim(local_features, global_features)
        logits = torch.cat((logits, nega.reshape(-1, 1)), dim=1)
        logits /= self.temperature
        labels = torch.zeros(local_features.size(0)).to(self.device).long()

        return self.ce_criterion(logits, labels)

    def get_perFCL_loss(
        self,
        local_features: torch.Tensor,
        old_local_features: torch.Tensor,
        global_features: torch.Tensor,
        old_global_features: torch.Tensor,
        aggregated_global_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        PerFCL loss consists of two contrastive losses.
        First one airms to enhance the similarity between the current global features and aggregated global feature
        as positive pairs while reducing the similarity between the current global features and old global
        features as negative pairs.
        Second one airms to enhance the similarity between the current local features and old local feature
        as positive pairs while reducing the similarity between the current local features and aggregated global
        features as negative pairs.
        """

        assert len(global_features) == len(aggregated_global_features)
        posi = self.cos_sim(global_features, aggregated_global_features)
        logits_max = posi.reshape(-1, 1)
        assert len(global_features) == len(old_global_features)
        nega = self.cos_sim(global_features, old_global_features)
        logits_max = torch.cat((logits_max, nega.reshape(-1, 1)), dim=1)
        logits_max /= self.temperature
        labels_max = torch.zeros(local_features.size(0)).to(self.device).long()

        assert len(local_features) == len(old_local_features)
        posi = self.cos_sim(local_features, old_local_features)
        logits_min = posi.reshape(-1, 1)
        assert len(local_features) == len(aggregated_global_features)
        nega = self.cos_sim(local_features, aggregated_global_features)
        logits_min = torch.cat((logits_min, nega.reshape(-1, 1)), dim=1)
        logits_min /= self.temperature
        labels_min = torch.zeros(local_features.size(0)).to(self.device).long()

        return self.ce_criterion(logits_min, labels_min), self.ce_criterion(logits_max, labels_max)

    def compute_loss(self, preds: Dict[str, torch.Tensor], target: torch.Tensor) -> Losses:
        if self.old_global_module is None:
            return super().compute_loss(preds, target)
        loss = self.criterion(preds["prediction"], target)

        # Optimal cos_sim_loss_weight is 100.0
        if self.cos_sim_loss_weight != 0.0:
            cos_loss = self.get_cosine_similarity_loss(
                local_features=preds["local_features"],
                global_features=preds["global_features"],
            )
            total_loss = loss + self.cos_sim_loss_weight * cos_loss
            losses = Losses(checkpoint=loss, backward=total_loss, additional_losses={"cos_sim_loss": cos_loss})

        # Optimal contrastive_loss_weight is 10.0
        elif self.contrastive_loss_weight:
            contrastive_loss = self.get_contrastive_loss(
                local_features=preds["local_features"],
                old_local_features=preds["old_local_features"],
                global_features=preds["global_features"],
            )
            total_loss = loss + self.contrastive_loss_weight * contrastive_loss
            losses = Losses(
                checkpoint=loss, backward=total_loss, additional_losses={"contrastive_loss": contrastive_loss}
            )

        # Optimal perFCL_loss_weight is [10.0, 10.0]
        elif self.perFCL_loss_weights[0] != 0.0 or self.perFCL_loss_weights[1] != 0.0:
            contrastive_loss_minimize, contrastive_loss_maximize = self.get_perFCL_loss(
                local_features=preds["local_features"],
                old_local_features=preds["old_local_features"],
                global_features=preds["global_features"],
                old_global_features=preds["old_global_features"],
                aggregated_global_features=preds["aggregated_global_features"],
            )
            total_loss = (
                loss
                + self.perFCL_loss_weights[0] * contrastive_loss_minimize
                + self.perFCL_loss_weights[1] * contrastive_loss_maximize
            )
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
