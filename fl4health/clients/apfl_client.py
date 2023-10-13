import copy
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

import torch
from flwr.common.typing import Config
from torch.optim import Optimizer

from fl4health.checkpointing.checkpointer import TorchCheckpointer
from fl4health.clients.basic_client import BasicClient
from fl4health.model_bases.apfl_base import APFLModule
from fl4health.parameter_exchange.layer_exchanger import FixedLayerExchanger
from fl4health.utils.losses import Losses, LossMeterType
from fl4health.utils.metrics import Metric, MetricMeterType


class ApflClient(BasicClient):
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
        # Apfl Module which holds both local and global models
        # and gives the ability to get personal, local and global predictions
        self.model: APFLModule

        # local_optimizer is used on the local model
        # Usual self.optimizer is used for global model
        self.local_optimizer: Optimizer

    def is_start_of_local_training(self, step: int) -> bool:
        return step == 0

    def update_after_step(self, step: int) -> None:
        if self.is_start_of_local_training(step) and self.model.adaptive_alpha:
            self.model.update_alpha()

    def split_optimizer(self, global_optimizer: Optimizer) -> Tuple[Optimizer, Optimizer]:
        """
        The optimizer from get_optimizer is for the entire APFLModule. We need one optimizer
        for the local model and one optimizer for the global model.
        """
        global_optimizer.param_groups.clear()
        global_optimizer.state.clear()
        local_optimizer = copy.deepcopy(global_optimizer)

        global_optimizer.add_param_group({"params": [p for p in self.model.global_model.parameters()]})
        local_optimizer.add_param_group({"params": [p for p in self.model.local_model.parameters()]})
        return global_optimizer, local_optimizer

    def setup_client(self, config: Config) -> None:
        """
        Set dataloaders, optimizers, parameter exchangers and other attributes derived from these.
        """
        super().setup_client(config)

        # Split optimizer from get_optimizer into two distinct optimizers
        # One for local model and one for global model
        global_optimizer, local_optimizer = self.split_optimizer(self.optimizer)
        self.optimizer = global_optimizer
        self.local_optimizer = local_optimizer

    def train_step(
        self, input: torch.Tensor, target: torch.Tensor
    ) -> Union[Tuple[Losses, torch.Tensor], Tuple[Losses, Dict[str, torch.Tensor]]]:
        # Return preds value of torch.Tensor containing personal, global and local predictions

        # Mechanics of training loop follow from original implementation
        # https://github.com/MLOPTPSU/FedTorch/blob/main/fedtorch/comms/trainings/federated/apfl.py

        # Forward pass on global model and update global parameters
        self.optimizer.zero_grad()
        global_pred = self.model.global_forward(input)
        global_loss = self.criterion(global_pred, target)
        global_loss.backward()
        self.optimizer.step()

        # Make sure gradients are zero prior to foward passes of global and local model
        # to generate personalized predictions
        # NOTE: We zero the global optimizer grads because they are used (after the backward calculation below)
        # to update the scalar alpha (see update_alpha() where .grad is called.)
        self.optimizer.zero_grad()
        self.local_optimizer.zero_grad()

        # Personal predictions are generated as a convex combination of the output
        # of local and global models
        preds = self.predict(input)
        assert isinstance(preds, dict)
        # Parameters of local model are updated to minimize loss of personalized model
        losses = self.compute_loss(preds, target)
        losses.backward.backward()
        self.local_optimizer.step()

        # Return dictionairy of predictions where key is used to name respective MetricMeters
        return losses, preds

    def get_parameter_exchanger(self, config: Config) -> FixedLayerExchanger:
        return FixedLayerExchanger(self.model.layers_to_exchange())

    def compute_loss(self, preds: Union[torch.Tensor, Dict[str, torch.Tensor]], target: torch.Tensor) -> Losses:
        assert isinstance(preds, dict)
        personal_loss = self.criterion(preds["personal"], target)
        global_loss = self.criterion(preds["global"], target)
        local_loss = self.criterion(preds["local"], target)
        additional_losses = {"global": global_loss, "local": local_loss}
        losses = Losses(checkpoint=personal_loss, backward=personal_loss, additional_losses=additional_losses)
        return losses
