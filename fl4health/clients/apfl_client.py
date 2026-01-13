from collections.abc import Sequence
from pathlib import Path

import torch
from flwr.common.typing import Config
from torch.optim import Optimizer

from fl4health.checkpointing.client_module import ClientCheckpointAndStateModule
from fl4health.clients.basic_client import BasicClient
from fl4health.metrics.base_metrics import Metric
from fl4health.model_bases.apfl_base import ApflModule
from fl4health.parameter_exchange.layer_exchanger import FixedLayerExchanger
from fl4health.reporting.base_reporter import BaseReporter
from fl4health.utils.losses import LossMeterType, TrainingLosses
from fl4health.utils.typing import TorchFeatureType, TorchInputType, TorchPredType, TorchTargetType


class ApflClient(BasicClient):
    def __init__(
        self,
        data_path: Path,
        metrics: Sequence[Metric],
        device: torch.device,
        loss_meter_type: LossMeterType = LossMeterType.AVERAGE,
        checkpoint_and_state_module: ClientCheckpointAndStateModule | None = None,
        reporters: Sequence[BaseReporter] | None = None,
        progress_bar: bool = False,
        client_name: str | None = None,
    ) -> None:
        """
        Client specifically implementing the APFL Algorithm: https://arxiv.org/abs/2003.13461.

        Twin models are trained. One of them is globally shared by all clients and aggregated on the server.
        The other is strictly trained locally by each client. Predictions are made by a convex combination of the
        models.

        Args:
            data_path (Path): path to the data to be used to load the data for client-side training.
            metrics (Sequence[Metric]): Metrics to be computed based on the labels and predictions of the client model.
            device (torch.device): Device indicator for where to send the model, batches, labels etc. Often "cpu" or
                "cuda".
            loss_meter_type (LossMeterType, optional): Type of meter used to track and compute the losses over
                each batch. Defaults to ``LossMeterType.AVERAGE``.
            checkpoint_and_state_module (ClientCheckpointAndStateModule | None, optional): A module meant to handle
                both checkpointing and state saving. The module, and its underlying model and state checkpointing
                components will determine when and how to do checkpointing during client-side training.
                No checkpointing (state or model) is done if not provided. Defaults to None.
            reporters (Sequence[BaseReporter] | None, optional): A sequence of FL4Health reporters which the client
                should send data to. Defaults to None.
            progress_bar (bool, optional): Whether or not to display a progress bar during client training and
                validation. Uses ``tqdm``. Defaults to False
            client_name (str | None, optional): An optional client name that uniquely identifies a client.
                If not passed, a hash is randomly generated. Client state will use this as part of its state file
                name. Defaults to None.
        """
        super().__init__(
            data_path=data_path,
            metrics=metrics,
            device=device,
            loss_meter_type=loss_meter_type,
            checkpoint_and_state_module=checkpoint_and_state_module,
            reporters=reporters,
            progress_bar=progress_bar,
            client_name=client_name,
        )

        self.model: ApflModule
        self.learning_rate: float
        self.optimizers: dict[str, torch.optim.Optimizer]

    def is_start_of_local_training(self, step: int) -> bool:
        return step == 0

    def update_after_step(self, step: int, current_round: int | None = None) -> None:
        """
        Called after local train step on client. Step is an integer that represents the local training step that was
        most recently completed.
        """
        if self.is_start_of_local_training(step) and self.model.adaptive_alpha:
            self.model.update_alpha()

    def train_step(self, input: TorchInputType, target: TorchTargetType) -> tuple[TrainingLosses, TorchPredType]:
        # Return preds value thats Dict of torch.Tensor containing personal, global and local predictions

        # Mechanics of training loop follow from original implementation
        # https://github.com/MLOPTPSU/FedTorch/blob/main/fedtorch/comms/trainings/federated/apfl.py

        # Forward pass on global model and update global parameters
        assert isinstance(input, torch.Tensor)
        self.optimizers["global"].zero_grad()
        global_pred = self.model.global_forward(input)
        global_loss = self.criterion(global_pred, target)
        global_loss.backward()
        self.optimizers["global"].step()

        # Make sure gradients are zero prior to forward passes of global and local model
        # to generate personalized predictions
        # NOTE: We zero the global optimizer grads because they are used (after the backward calculation below)
        # to update the scalar alpha (see update_alpha() where .grad is called.)
        self.optimizers["global"].zero_grad()
        self.optimizers["local"].zero_grad()

        # Personal predictions are generated as a convex combination of the output
        # of local and global models
        preds, features = self.predict(input)
        target = self.transform_target(target)  # Apply transformation (Defaults to identity)

        # Parameters of local model are updated to minimize loss of personalized model
        losses = self.compute_training_loss(preds, features, target)

        losses.backward["backward"].backward()
        self.optimizers["local"].step()

        # Return dictionary of predictions where key is used to name respective MetricMeters
        return losses, preds

    def get_parameter_exchanger(self, config: Config) -> FixedLayerExchanger:
        return FixedLayerExchanger(self.model.layers_to_exchange())

    def compute_loss_and_additional_losses(
        self,
        preds: TorchPredType,
        features: TorchFeatureType,
        target: TorchTargetType,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Computes the loss and any additional losses given predictions of the model and ground truth data.
        For APFL, the loss will be the personal loss and the additional losses are the global and local loss.

        Args:
            preds (TorchPredType): Prediction(s) of the model(s) indexed by name.
            features (TorchFeatureType): Feature(s) of the model(s) indexed by name.
            target (TorchTargetType): Ground truth data to evaluate predictions against.

        Returns:
            (tuple[torch.Tensor, dict[str, torch.Tensor]]): A tuple with:

                - The tensor for the personal loss
                - A dictionary of with "global_loss" and "local_loss" keys and their calculated values
        """
        assert isinstance(preds, dict)
        personal_loss = self.criterion(preds["personal"], target)
        global_loss = self.criterion(preds["global"], target)
        local_loss = self.criterion(preds["local"], target)
        additional_losses = {"global": global_loss, "local": local_loss}

        return personal_loss, additional_losses

    def set_optimizer(self, config: Config) -> None:
        optimizers = self.get_optimizer(config)
        assert isinstance(optimizers, dict) and {"global", "local"} == set(optimizers.keys())
        self.optimizers = optimizers

    def get_optimizer(self, config: Config) -> dict[str, Optimizer]:
        """Returns a dictionary with global and local optimizers with string keys "global" and "local" respectively."""
        raise NotImplementedError
