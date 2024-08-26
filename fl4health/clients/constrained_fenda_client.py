from logging import WARNING
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from flwr.common.logger import log
from flwr.common.typing import Config

from fl4health.checkpointing.client_module import ClientCheckpointModule
from fl4health.clients.fenda_client import FendaClient
from fl4health.losses.fenda_loss_config import ConstrainedFendaLossContainer
from fl4health.model_bases.fenda_base import FendaModelWithFeatureState
from fl4health.parameter_exchange.parameter_exchanger_base import ParameterExchanger
from fl4health.utils.losses import EvaluationLosses, LossMeterType
from fl4health.utils.metrics import Metric
from fl4health.utils.typing import TorchFeatureType, TorchInputType, TorchPredType, TorchTargetType


class ConstrainedFendaClient(FendaClient):
    def __init__(
        self,
        data_path: Path,
        metrics: Sequence[Metric],
        device: torch.device,
        loss_meter_type: LossMeterType = LossMeterType.AVERAGE,
        checkpointer: Optional[ClientCheckpointModule] = None,
        loss_container: Optional[ConstrainedFendaLossContainer] = None,
    ) -> None:
        """
        This class extends the functionality of FENDA training to include various kinds of constraints applied during
        the client-side training of FENDA models.

        Args:
            data_path (Path): path to the data to be used to load the data for client-side training
            metrics (Sequence[Metric]): Metrics to be computed based on the labels and predictions of the client model
            device (torch.device): Device indicator for where to send the model, batches, labels etc. Often 'cpu' or
                'cuda'
            loss_meter_type (LossMeterType, optional): Type of meter used to track and compute the losses over
                each batch. Defaults to LossMeterType.AVERAGE.
            checkpointer (Optional[ClientCheckpointModule], optional): Checkpointer module defining when and how to
                do checkpointing during client-side training. No checkpointing is done if not provided. Defaults to
                None.
            loss_configuration (Optional[ConstrainedFendaLossContainer], optional): Configuration that determines which
                losses will be applied during FENDA training. Defaults to None.
        """
        super().__init__(
            data_path=data_path,
            metrics=metrics,
            device=device,
            loss_meter_type=loss_meter_type,
            checkpointer=checkpointer,
        )

        if loss_container:
            self.loss_container = loss_container
        else:
            # If no loss configuration has been define, set everything to zero. This is equivalent to vanilla FENDA
            log(
                WARNING,
                "No loss container provided, defaulting to an empty container. "
                "This is equivalent to running a vanilla FENDA client",
            )
            self.loss_container = ConstrainedFendaLossContainer(None, None, None)

        # Need to save previous local module, global module and aggregated global module at each communication round
        # to compute contrastive loss.
        self.old_local_module: Optional[nn.Module] = None
        self.old_global_module: Optional[nn.Module] = None
        self.initial_global_module: Optional[nn.Module] = None

    def get_parameter_exchanger(self, config: Config) -> ParameterExchanger:
        assert isinstance(self.model, FendaModelWithFeatureState)
        return super().get_parameter_exchanger(config)

    def _flatten(self, features: torch.Tensor) -> torch.Tensor:
        """
        Flatten the provided features ASSUMING they are provided in batch-first format.

        Args:
            features (torch.Tensor): features to be flattened

        Returns:
            torch.Tensor: flattened feature vectors of shape (batch, -1)
        """
        return features.reshape(len(features), -1)

    def _perfcl_keys_present(self, features: Dict[str, torch.Tensor]) -> bool:
        target_keys = {
            "old_local_features",
            "old_global_features",
            "initial_global_features",
        }
        return target_keys.issubset(features.keys())

    def predict(self, input: TorchInputType) -> Tuple[TorchPredType, TorchFeatureType]:
        """
        Computes the prediction(s) and features of the model(s) given the input.

        Args:
            input (TorchInputType): Inputs to be fed into the model. TorchInputType is simply an alias
            for the union of torch.Tensor and Dict[str, torch.Tensor].

        Returns:
            Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]: A tuple in which the first element
            contains predictions indexed by name and the second element contains intermediate activations
            index by name. Specifically the features of the model, features of the global model and features of
            the old model are returned. All predictions included in dictionary will be used to compute metrics.
        """
        assert isinstance(input, torch.Tensor)
        assert isinstance(self.model, FendaModelWithFeatureState)
        preds, features = self.model(input)

        if self.loss_container.has_contrastive_loss() or self.loss_container.has_perfcl_loss():
            # If we have defined a contrastive loss function or PerFCL loss function, we attempt to save old local
            # features.
            if self.old_local_module is not None:
                features["old_local_features"] = self._flatten(self.old_local_module.forward(input))

        if self.loss_container.has_perfcl_loss():
            # If a PerFCL loss function has been defined, then we also save two additional feature components.
            if self.old_global_module is not None:
                features["old_global_features"] = self._flatten(self.old_global_module.forward(input))
            if self.initial_global_module is not None:
                features["initial_global_features"] = self._flatten(self.initial_global_module.forward(input))

        return preds, features

    def update_after_train(self, local_steps: int, loss_dict: Dict[str, float], config: Config) -> None:
        """
        This function is called after client-side training concludes. If a contrastive or PerFCL loss function has
        been defined, it is used to save the local and global feature extraction weights/modules to be used in the
        next round of client-side training.

        Args:
            local_steps (int): Number of steps performed during training
            loss_dict (Dict[str, float]): Losses computed during training.
        """
        # Save the parameters of the old model
        assert isinstance(self.model, FendaModelWithFeatureState)

        if self.loss_container.has_contrastive_loss() or self.loss_container.has_perfcl_loss():
            self.old_local_module = self.clone_and_freeze_model(self.model.first_feature_extractor)
            self.old_global_module = self.clone_and_freeze_model(self.model.second_feature_extractor)

        super().update_after_train(local_steps, loss_dict, config)

    def update_before_train(self, current_server_round: int) -> None:
        """
        This function is called prior to the start of client-side training, but after the server parameters have be
        received and injected into the model. If a PerFCL loss function has been defined, it is used to save the
        aggregated global feature extractor weights/module representing the initial state of this module BEFORE this
        iteration of client-side training but AFTER server-side aggregation.

        Args:
            current_server_round (int): Current server round being performed.
        """
        # Save the parameters of the aggregated global model
        assert isinstance(self.model, FendaModelWithFeatureState)

        if self.loss_container.has_perfcl_loss():
            self.initial_global_module = self.clone_and_freeze_model(self.model.second_feature_extractor)

        super().update_before_train(current_server_round)

    def compute_loss_and_additional_losses(
        self,
        preds: TorchPredType,
        features: TorchFeatureType,
        target: TorchTargetType,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Computes the loss and any additional losses given predictions of the model and ground truth data.
        For FENDA, the loss is the total loss and the additional losses are the loss, total loss and, based on
        client attributes set from server config, cosine similarity loss, contrastive loss and perfcl losses.

        Args:
            preds (Dict[str, torch.Tensor]): Prediction(s) of the model(s) indexed by name.
            features (Dict[str, torch.Tensor]): Feature(s) of the model(s) indexed by name.
            target (torch.Tensor): Ground truth data to evaluate predictions against.

        Returns:
            Tuple[torch.Tensor, Union[Dict[str, torch.Tensor], None]]; A tuple with:
                - The tensor for the total loss
                - A dictionary with `loss`, `total_loss` and, based on client attributes set from server config, also
                    `cos_sim_loss`, `contrastive_loss`, `contrastive_loss_minimize` and `contrastive_loss_minimize`
                    keys and their respective calculated values.
        """

        loss = self.criterion(preds["prediction"], target)
        total_loss = loss.clone()
        additional_losses = {"loss": loss}

        if self.loss_container.has_cosine_similarity_loss():
            cosine_similarity_loss = self.loss_container.compute_cosine_similarity_loss(
                features["local_features"], features["global_features"]
            )
            total_loss += cosine_similarity_loss
            additional_losses["cos_sim_loss"] = cosine_similarity_loss

        if self.loss_container.has_contrastive_loss() and "old_local_features" in features:
            contrastive_loss = self.loss_container.compute_contrastive_loss(
                features["local_features"],
                features["old_local_features"].unsqueeze(0),
                features["global_features"].unsqueeze(0),
            )
            total_loss += contrastive_loss
            additional_losses["contrastive_loss"] = contrastive_loss

        if self.loss_container.has_perfcl_loss() and self._perfcl_keys_present(features):
            global_feature_contrastive_loss, local_feature_contrastive_loss = self.loss_container.compute_perfcl_loss(
                features["local_features"],
                features["old_local_features"],
                features["global_features"],
                features["old_global_features"],
                features["initial_global_features"],
            )
            total_loss += global_feature_contrastive_loss + local_feature_contrastive_loss
            additional_losses["global_feature_contrastive_loss"] = global_feature_contrastive_loss
            additional_losses["local_feature_contrastive_loss"] = local_feature_contrastive_loss

        additional_losses["total_loss"] = total_loss

        return total_loss, additional_losses

    def compute_evaluation_loss(
        self,
        preds: TorchPredType,
        features: TorchFeatureType,
        target: TorchTargetType,
    ) -> EvaluationLosses:
        """
        Computes evaluation loss given predictions of the model and ground truth data. Optionally computes
        additional loss components such as cosine_similarity_loss, contrastive_loss and perfcl_loss based on
        client attributes set from server config.

        Args:
            preds (Dict[str, torch.Tensor]): Prediction(s) of the model(s) indexed by name.
                All predictions included in dictionary will be used to compute metrics.
            features: (Dict[str, torch.Tensor]): Feature(s) of the model(s) indexed by name.
            target: (torch.Tensor): Ground truth data to evaluate predictions against.

        Returns:
            EvaluationLosses: an instance of EvaluationLosses containing checkpoint loss and additional losses
                indexed by name. Additional losses may include cosine_similarity_loss, contrastive_loss
                and perfcl_loss.
        """
        _, additional_losses = self.compute_loss_and_additional_losses(preds, features, target)
        return EvaluationLosses(checkpoint=additional_losses["loss"], additional_losses=additional_losses)
