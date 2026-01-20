from collections.abc import Sequence
from logging import WARNING
from pathlib import Path

import torch
from flwr.common.logger import log

from fl4health.checkpointing.client_module import ClientCheckpointAndStateModule
from fl4health.clients.basic_client import BasicClient, Config
from fl4health.losses.contrastive_loss import MoonContrastiveLoss
from fl4health.metrics.base_metrics import Metric
from fl4health.model_bases.sequential_split_models import SequentiallySplitModel
from fl4health.reporting.base_reporter import BaseReporter
from fl4health.utils.client import clone_and_freeze_model
from fl4health.utils.losses import EvaluationLosses, LossMeterType, TrainingLosses
from fl4health.utils.typing import TorchFeatureType, TorchInputType, TorchPredType, TorchTargetType


class MoonClient(BasicClient):
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
        temperature: float = 0.5,
        contrastive_weight: float = 1.0,
        len_old_models_buffer: int = 1,
    ) -> None:
        """
        This client implements the MOON algorithm from Model-Contrastive Federated Learning. The key idea of MOON is
        to enforce similarity between representations from the global and current local model through a contrastive
        loss to constrain the local training of individual parties in the non-IID setting.

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
            temperature (float, optional): Temperature used in the calculation of the contrastive loss.
                Defaults to 0.5.
            contrastive_weight (float, optional): Weight placed on the contrastive loss function. Referred to as mu
                in the original paper. Defaults to 1.0.
            len_old_models_buffer (int, optional): Number of old models to be stored for computation in the
                contrastive learning loss function. Defaults to 1.
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
        self.temperature = temperature
        self.contrastive_weight = contrastive_weight
        if self.contrastive_weight == 0:
            log(WARNING, "Contrastive loss weight is set to 0, thus Contrastive loss will not be computed.")
        self.contrastive_loss_function = MoonContrastiveLoss(self.device, temperature=temperature)

        # Saving previous local models and a global model at each communication round to compute contrastive loss
        self.len_old_models_buffer = len_old_models_buffer
        self.old_models_list: list[torch.nn.Module] = []
        self.global_model: torch.nn.Module | None = None

    def predict(self, input: TorchInputType) -> tuple[TorchPredType, TorchFeatureType]:
        """
        Computes the prediction(s) and features of the model(s) given the input. This function also produces the
        necessary features from the ``global_model`` (aggregated model from server) and ``old_models`` (previous
        client-side optimized models) in order to be able to compute the appropriate contrastive loss.

        Args:
            input (TorchInputType): Inputs to be fed into the model. ``TorchInputType`` is simply an alias
                for the union of ``torch.Tensor`` and ``dict[str, torch.Tensor]``. Here, the MOON models require
                input to simply be of type ``torch.Tensor``

        Returns:
            (tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]): A tuple in which the first element
                contains predictions indexed by name and the second element contains intermediate activations
                index by name. Specifically the features of the model, features of the global model and features of
                the old model are returned. All predictions included in dictionary will be used to compute metrics.
        """
        assert isinstance(input, torch.Tensor)
        preds, features = self.model(input)
        assert "features" in features, "Model must produce a features dictionary with a 'features' key"

        # If there are no models in the old_models_list, we don't compute the features for the contrastive loss
        if len(self.old_models_list) > 0:
            # Features from each of the old models are packed into a single tensor
            old_features = torch.zeros(len(self.old_models_list), *features["features"].size()).to(self.device)
            for i, old_model in enumerate(self.old_models_list):
                _, old_model_features = old_model(input)
                old_features[i] = old_model_features["features"]
            features.update({"old_features": old_features})

        if self.global_model is not None:
            _, global_model_features = self.global_model(input)
            features.update({"global_features": global_model_features["features"]})
        return preds, features

    def update_after_train(self, local_steps: int, loss_dict: dict[str, float], config: Config) -> None:
        """
        This function is called immediately after client-side training has completed. This function saves the final
        trained model to the list of old models to be used in subsequent server rounds.

        Args:
            local_steps (int): Number of local steps performed during training
            loss_dict (dict[str, float]): Loss dictionary associated with training.
            config (Config): The config from the server
        """
        assert isinstance(self.model, SequentiallySplitModel)
        # Save the parameters of the old LOCAL model
        old_model = clone_and_freeze_model(self.model)
        # Current model is appended to the back of the list
        self.old_models_list.append(old_model)
        # If the list is longer than desired, the element at the front of the list is removed.
        if len(self.old_models_list) > self.len_old_models_buffer:
            self.old_models_list.pop(0)

        super().update_after_train(local_steps, loss_dict, config)

    def update_before_train(self, current_server_round: int) -> None:
        """
        This function is called before training, immediately after injecting the aggregated server weights into the
        client model. We clone and free the current model to preserve the aggregated server weights state (i.e. the
        initial model before training starts).

        Args:
            current_server_round (int): Current federated training round being executed.
        """
        # Save the parameters of the global model
        self.global_model = clone_and_freeze_model(self.model)

        super().update_before_train(current_server_round)

    def compute_loss_and_additional_losses(
        self,
        preds: TorchPredType,
        features: TorchFeatureType,
        target: TorchTargetType,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Computes the loss and any additional losses given predictions of the model and ground truth data.
        For MOON, the loss is the total loss (criterion and weighted contrastive loss) and the additional losses are
        the loss, (unweighted) contrastive loss, and total loss.

        Args:
            preds (TorchPredType): Prediction(s) of the model(s) indexed by name.
            features (TorchFeatureType): Feature(s) of the model(s) indexed by name.
            target (TorchTargetType): Ground truth data to evaluate predictions against.

        Returns:
            (tuple[torch.Tensor, dict[str, torch.Tensor]]): A tuple with:

                - The tensor for the total loss
                - A dictionary with ``loss``, ``contrastive_loss`` and ``total_loss`` keys and their calculated values.
        """
        loss = self.criterion(preds["prediction"], target)
        total_loss = loss.clone()
        additional_losses = {
            "loss": loss,
        }

        if "old_features" in features and "global_features" in features:
            contrastive_loss = self.contrastive_loss_function(
                features["features"], features["global_features"].unsqueeze(0), features["old_features"]
            )
            total_loss += self.contrastive_weight * contrastive_loss
            additional_losses["contrastive_loss"] = contrastive_loss

        additional_losses["total_loss"] = total_loss

        return total_loss, additional_losses

    def compute_training_loss(
        self,
        preds: TorchPredType,
        features: TorchFeatureType,
        target: TorchTargetType,
    ) -> TrainingLosses:
        """
        Computes training loss given predictions and features of the model and ground truth data. Loss includes
        base loss plus a model contrastive loss.

        Args:
            preds (TorchPredType): Prediction(s) of the model(s) indexed by name.
                All predictions included in dictionary will be used to compute metrics.
            features (dict[str, torch.Tensor]): Feature(s) of the model(s) indexed by name.
            target (torch.Tensor): Ground truth data to evaluate predictions against.

        Returns:
            (TrainingLosses): An instance of ``TrainingLosses`` containing backward loss and additional losses
                indexed by name.
        """
        # Check that the model is in training mode
        assert self.model.training

        # If there are no old local models in the list (first pass of MOON training), we just do basic loss
        #  calculations
        if len(self.old_models_list) == 0:
            total_loss, additional_losses = super().compute_loss_and_additional_losses(preds, features, target)
        else:
            total_loss, additional_losses = self.compute_loss_and_additional_losses(preds, features, target)

        return TrainingLosses(backward=total_loss, additional_losses=additional_losses)

    def compute_evaluation_loss(
        self,
        preds: TorchPredType,
        features: TorchFeatureType,
        target: TorchTargetType,
    ) -> EvaluationLosses:
        """
        Computes evaluation loss given predictions and features of the model and ground truth data. Loss includes
        base loss plus a model contrastive loss.

        Args:
            preds (TorchPredType): Prediction(s) of the model(s) indexed by name.
                All predictions included in dictionary will be used to compute metrics.
            features (dict[str, torch.Tensor]): Feature(s) of the model(s) indexed by name.
            target (torch.Tensor): Ground truth data to evaluate predictions against.

        Returns:
            (EvaluationLosses): An instance of ``EvaluationLosses`` containing checkpoint loss and additional losses
                indexed by name.
        """
        # Check that the model is in evaluation mode
        assert not self.model.training

        # If there are no old local models in the list (first pass of MOON training), we just do basic loss
        # calculations
        if len(self.old_models_list) == 0:
            checkpoint_loss, additional_losses = super().compute_loss_and_additional_losses(preds, features, target)
        else:
            _, additional_losses = self.compute_loss_and_additional_losses(preds, features, target)
            # Note that the first parameter returned is the "total loss", which includes the contrastive loss
            # So we use the vanilla loss stored in additional losses for checkpointing.
            checkpoint_loss = additional_losses["loss"]
        return EvaluationLosses(checkpoint=checkpoint_loss, additional_losses=additional_losses)
