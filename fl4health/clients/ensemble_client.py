from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import torch
from flwr.common.typing import Config
from torch.optim import Optimizer

from fl4health.checkpointing.checkpointer import TorchCheckpointer
from fl4health.clients.basic_client import BasicClient
from fl4health.model_bases.ensemble_base import EnsembleModel
from fl4health.utils.losses import Losses, LossMeterType
from fl4health.utils.metrics import Metric


class EnsembleClient(BasicClient):
    def __init__(
        self,
        data_path: Path,
        metrics: Sequence[Metric],
        device: torch.device,
        loss_meter_type: LossMeterType = LossMeterType.AVERAGE,
        checkpointer: Optional[TorchCheckpointer] = None,
    ) -> None:
        super().__init__(
            data_path=data_path,
            metrics=metrics,
            device=device,
            loss_meter_type=loss_meter_type,
            checkpointer=checkpointer,
        )

        self.model: EnsembleModel

    def setup_client(self, config: Config) -> None:
        super().setup_client(config)

        assert len(self.optimizers) == len(self.model.models)
        assert all(
            opt_key == model_key
            for opt_key, model_key in zip(sorted(self.optimizers.keys()), sorted(self.model.models.keys()))
        )

    def set_optimizer(self, config: Config) -> None:
        optimizers = self.get_optimizer(config)
        assert isinstance(optimizers, dict)
        self.optimizers = optimizers

    def train_step(self, input: torch.Tensor, target: torch.Tensor) -> Tuple[Losses, Dict[str, torch.Tensor]]:

        for optimizer in self.optimizers.values():
            optimizer.zero_grad()

        preds, features = self.predict(input)
        losses = self.compute_loss(preds, features, target)
        losses.backward.backward()

        for optimizer in self.optimizers.values():
            optimizer.step()

        return losses, preds

    def compute_loss(
        self, preds: Dict[str, torch.Tensor], features: Dict[str, torch.Tensor], target: torch.Tensor
    ) -> Losses:
        loss_dict = {}
        for key, pred in preds.items():
            loss_dict[key] = self.criterion(pred, target)

        individual_model_loss_list = {key: loss for key, loss in loss_dict.items() if key != "ensemble-pred"}
        backward_loss = torch.sum(torch.tensor(list(individual_model_loss_list.values()), requires_grad=True))
        checkpoint_loss = loss_dict["ensemble-pred"]
        additional_losses = individual_model_loss_list

        losses = Losses(checkpoint=checkpoint_loss, backward=backward_loss, additional_losses=additional_losses)
        return losses

    def get_optimizer(self, config: Config) -> Dict[str, Optimizer]:
        raise NotImplementedError
