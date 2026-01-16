from collections.abc import Sequence
from pathlib import Path

import torch
from flwr.common.typing import Config
from torch.optim import Optimizer

from fl4health.checkpointing.client_module import ClientCheckpointAndStateModule
from fl4health.clients.basic_client import BasicClient
from fl4health.metrics.base_metrics import Metric
from fl4health.model_bases.ensemble_base import EnsembleModel
from fl4health.reporting.base_reporter import BaseReporter
from fl4health.utils.losses import EvaluationLosses, LossMeterType, TrainingLosses
from fl4health.utils.typing import TorchFeatureType, TorchInputType, TorchPredType, TorchTargetType


class EnsembleClient(BasicClient):
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
        This client enables the training of ensemble models in a federated manner.

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

        self.model: EnsembleModel

    def setup_client(self, config: Config) -> None:
        """
        Set dataloaders, optimizers, parameter exchangers and other attributes derived from these.
        Then set initialized attribute to True. Also perform some checks to ensure the keys of the
        optimizer dictionary are consistent with the model keys.

        Args:
            config (Config): The config from the server.
        """
        super().setup_client(config)

        assert len(self.optimizers) == len(self.model.ensemble_models)
        assert all(
            opt_key == model_key
            for opt_key, model_key in zip(sorted(self.optimizers.keys()), sorted(self.model.ensemble_models.keys()))
        )

    def set_optimizer(self, config: Config) -> None:
        """
        Method called in the the ``setup_client`` method to set optimizer attribute returned by used-defined
        ``get_optimizer``. Ensures that the return value of ``get_optimizer`` is a dictionary since that is required
        for the ensemble client.

        Args:
            config (Config): The config from the server.
        """
        optimizers = self.get_optimizer(config)
        assert isinstance(optimizers, dict)
        self.optimizers = optimizers

    def train_step(self, input: TorchInputType, target: TorchTargetType) -> tuple[TrainingLosses, TorchPredType]:
        """
        Given a single batch of input and target data, generate predictions (both individual models and ensemble
        prediction), compute loss, update parameters and optionally update metrics if they exist. (i.e.
        backpropagation on a single batch of data). Assumes ``self.model`` is in train mode already. Differs from
        parent  method in that, there are multiple losses that we have to do backward passes on and multiple
        optimizers to  update parameters each train step.

        Args:
            input (TorchInputType): The input to be fed into the model. ``TorchInputType`` is simply an alias for the
                union of ``torch.Tensor`` and ``dict[str, torch.Tensor]``.
            target (TorchTargetType): The target corresponding to the input.

        Returns:
            (tuple[TrainingLosses, dict[str, torch.Tensor]]): The losses object from the train step along with
                a dictionary of any predictions produced by the model.
        """
        assert isinstance(input, torch.Tensor)
        for optimizer in self.optimizers.values():
            optimizer.zero_grad()

        preds, features = self.predict(input)
        target = self.transform_target(target)  # Apply transformation (Defaults to identity)

        losses = self.compute_training_loss(preds, features, target)

        for loss in losses.backward.values():
            loss.backward()

        for optimizer in self.optimizers.values():
            optimizer.step()

        return losses, preds

    def compute_training_loss(
        self,
        preds: TorchPredType,
        features: TorchFeatureType,
        target: TorchTargetType,
    ) -> TrainingLosses:
        """
        Computes training loss given predictions (and potentially features) of the model and ground truth data.
        Since the ensemble client has more than one model, there are multiple backward losses that exist.

        Args:
            preds (TorchPredType): Prediction(s) of the model(s) indexed by name. Anything stored
                in preds will be used to compute metrics.
            features (dict[str, torch.Tensor]): Feature(s) of the model(s) indexed by name.
            target (torch.Tensor): Ground truth data to evaluate predictions against.

        Returns:
            (TrainingLosses): An instance of ``TrainingLosses`` containing backward loss and additional losses
                indexed by name.
        """
        loss_dict = {}
        for key, pred in preds.items():
            loss_dict[key] = self.criterion(pred.float(), target)

        individual_model_losses = {key: loss for key, loss in loss_dict.items() if key != "ensemble-pred"}
        return TrainingLosses(backward=individual_model_losses)

    def compute_evaluation_loss(
        self,
        preds: TorchPredType,
        features: TorchFeatureType,
        target: TorchTargetType,
    ) -> EvaluationLosses:
        """
        Computes evaluation loss given predictions (and potentially features) of the model and ground truth data.
        Since the ensemble client has more than one model, there are multiple backward losses that exist.

        Args:
            preds (TorchPredType): Prediction(s) of the model(s) indexed by name. Anything stored
                in preds will be used to compute metrics.
            features (dict[str, torch.Tensor]): Feature(s) of the model(s) indexed by name.
            target (torch.Tensor): Ground truth data to evaluate predictions against.

        Returns:
            (EvaluationLosses): An instance of ``EvaluationLosses`` containing checkpoint loss and additional losses
                indexed by name.
        """
        loss_dict = {}
        for key, pred in preds.items():
            loss_dict[key] = self.criterion(pred.float(), target)

        checkpoint_loss = loss_dict["ensemble-pred"]
        return EvaluationLosses(checkpoint=checkpoint_loss)

    def get_optimizer(self, config: Config) -> dict[str, Optimizer]:
        """
        Method to be defined by user that returns dictionary of optimizers with keys corresponding to the keys of the
        models in ``EnsembleModel`` that the optimizer applies too.

        Args:
            config (Config): The config sent from the server.

        Returns:
            (dict[str, Optimizer]): An optimizer or dictionary of optimizers to train model.

        Raises:
            NotImplementedError: To be defined in child class.
        """
        raise NotImplementedError
