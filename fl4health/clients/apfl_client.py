from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

import torch
from flwr.common.typing import Config
from torch.optim import Optimizer

from fl4health.checkpointing.checkpointer import TorchCheckpointer
from fl4health.clients.basic_client import BasicClient
from fl4health.model_bases.apfl_base import ApflModule
from fl4health.parameter_exchange.layer_exchanger import FixedLayerExchanger
from fl4health.utils.losses import Losses, LossMeterType
from fl4health.utils.metrics import Metric


class ApflClient(BasicClient):
    def __init__(
        self,
        data_path: Path,
        metrics: Sequence[Metric],
        device: torch.device,
        loss_meter_type: LossMeterType = LossMeterType.AVERAGE,
        checkpointer: Optional[TorchCheckpointer] = None,
    ) -> None:
        super().__init__(data_path, metrics, device, loss_meter_type, checkpointer)

        self.model: ApflModule
        self.learning_rate: float
        self.optimizer: torch.optim.Optimizer
        self.local_optimizer: torch.optim.Optimizer

    def is_start_of_local_training(self, step: int) -> bool:
        return step == 0

    def update_after_step(self, step: int) -> None:
        """
        Called after local train step on client. step is an integer that represents
        the local training step that was most recently completed.
        """
        if self.is_start_of_local_training(step) and self.model.adaptive_alpha:
            self.model.update_alpha()

    def train_step(self, input: torch.Tensor, target: torch.Tensor) -> Tuple[Losses, Dict[str, torch.Tensor]]:
        # Return preds value thats Dict of torch.Tensor containing personal, global and local predictions

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
        preds, features = self.predict(input)
        # Parameters of local model are updated to minimize loss of personalized model
        losses = self.compute_loss(preds, features, target)
        losses.backward.backward()
        self.local_optimizer.step()

        # Return dictionary of predictions where key is used to name respective MetricMeters
        return losses, preds

    def get_parameter_exchanger(self, config: Config) -> FixedLayerExchanger:
        return FixedLayerExchanger(self.model.layers_to_exchange())

    def compute_loss(
        self,
        preds: Union[torch.Tensor, Dict[str, torch.Tensor]],
        features: Dict[str, torch.Tensor],
        target: torch.Tensor,
    ) -> Losses:
        """
        Computes loss given predictions of the model and ground truth data.

        Args:
            preds (Dict[str, torch.Tensor]): Prediction(s) of the model(s) indexed by name.
            features: (Dict[str, torch.Tensor]): Feature(s) of the model(s) indexed by name.
            target: (torch.Tensor): Ground truth data to evaluate predictions against.

        Returns:
            Losses: Object containing checkpoint loss, backward loss and additional losses indexed by name.
            Additional losses include global and local losses.
        """
        assert isinstance(preds, dict)
        personal_loss = self.criterion(preds["personal"], target)
        global_loss = self.criterion(preds["global"], target)
        local_loss = self.criterion(preds["local"], target)
        additional_losses = {"global": global_loss, "local": local_loss}
        losses = Losses(checkpoint=personal_loss, backward=personal_loss, additional_losses=additional_losses)
        return losses

    def set_optimizer(self, config: Config) -> None:
        optimizer_dict = self.get_optimizer(config)
        assert isinstance(optimizer_dict, dict)
        self.optimizer = optimizer_dict["global"]
        self.local_optimizer = optimizer_dict["local"]

    def get_optimizer(self, config: Config) -> Dict[str, Optimizer]:
        """
        Returns a dictionary with global and local optimizers with string keys 'global' and 'local' respectively.
        """
        raise NotImplementedError
