from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import torch
from flwr.common.typing import Config

from fl4health.checkpointing.client_module import ClientCheckpointModule
from fl4health.clients.basic_client import BasicClient
from fl4health.losses.perfcl_loss import PerFclLoss
from fl4health.model_bases.perfcl_base import PerFclModel
from fl4health.parameter_exchange.layer_exchanger import FixedLayerExchanger
from fl4health.parameter_exchange.parameter_exchanger_base import ParameterExchanger
from fl4health.utils.losses import EvaluationLosses, LossMeterType
from fl4health.utils.metrics import Metric
from fl4health.utils.typing import TorchFeatureType, TorchInputType, TorchPredType, TorchTargetType


class PerFclClient(BasicClient):
    def __init__(
        self,
        data_path: Path,
        metrics: Sequence[Metric],
        device: torch.device,
        loss_meter_type: LossMeterType = LossMeterType.AVERAGE,
        checkpointer: Optional[ClientCheckpointModule] = None,
        global_feature_loss_temperature: float = 0.5,
        local_feature_loss_temperature: float = 0.5,
        global_feature_contrastive_loss_weight: float = 1.0,
        local_feature_contrastive_loss_weight: float = 1.0,
    ) -> None:
        """
        This client is used to perform client-side training associated with the PerFCL method derived in
        https://www.sciencedirect.com/science/article/pii/S0031320323002078. The approach attempts to manipulate the
        training dynamics of a parallel weight split model with a global feature extractor, that is aggregated
        on the server-side with FedAvg and a local feature extractor that is only locally trained. This method is
        related to FENDA, but with additional losses on the latent spaces of the local and global feature extractors.

        Args:
            data_path (Path): Path to the data directory.
            metrics (Sequence[Metric]): List of metrics to be used for evaluation.
            device (torch.device): Device to be used for training.
            loss_meter_type (LossMeterType, optional): Type of loss meter to be used. Defaults to
                LossMeterType.AVERAGE.
            checkpointer (Optional[ClientCheckpointModule], optional): Checkpointer module defining when and how to
                do checkpointing during client-side training. No checkpointing is done if not provided. Defaults to
                None.
            global_feature_loss_temperature (float, optional): Temperature to be used in the contrastive loss
                associated with constraining the global feature extractor in the PerFCL loss. Defaults to 0.5.
            local_feature_loss_temperature (float, optional): Temperature to be used in the contrastive loss
                associated with constraining the local feature extractor in the PerFCL loss. Defaults to 0.5.
            global_feature_contrastive_loss_weight (float, optional): Weight on the contrastive loss value associated
                with the global feature extractor. REFERRED TO AS MU in the original paper. Defaults to 1.0.
            local_feature_contrastive_loss_weight (float, optional): Weight on the contrastive loss value associated
                with the local feature extractor. REFERRED TO AS GAMMA in the original paper.  Defaults to 1.0.
        """
        super().__init__(
            data_path=data_path,
            metrics=metrics,
            device=device,
            loss_meter_type=loss_meter_type,
            checkpointer=checkpointer,
        )
        self.global_feature_contrastive_loss_weight = global_feature_contrastive_loss_weight
        self.local_feature_contrastive_loss_weight = local_feature_contrastive_loss_weight
        self.perfcl_loss_function = PerFclLoss(
            self.device, global_feature_loss_temperature, local_feature_loss_temperature
        )

        # In order to compute the PerFCL losses, we need to save final local module and global modules from the
        # previous iteration of client-side training and initial global module passed to the client after server-side
        # aggregation at each communication round
        self.old_local_module: Optional[torch.nn.Module] = None
        self.old_global_module: Optional[torch.nn.Module] = None
        self.initial_global_module: Optional[torch.nn.Module] = None

    def get_parameter_exchanger(self, config: Config) -> ParameterExchanger:
        """
        Sets the parameter exchanger to be used by the clients to send parameters to and receive them from the server
        For PerFCL clients, a FixedLayerExchanger is used by default. We also required that the model being exchanged
        is of the PerFclModel type to ensure that the appropriate layers are exchanged.

        Args:
            config (Config): Configuration provided by the server.

        Returns:
            ParameterExchanger: FixedLayerExchanger meant to only exchange a subset of model layers with the server
                for aggregation.
        """
        assert isinstance(self.model, PerFclModel)
        return FixedLayerExchanger(self.model.layers_to_exchange())

    def _flatten(self, features: torch.Tensor) -> torch.Tensor:
        """
        Flatten the provided features ASSUMING they are provided in batch-first format.

        Args:
            features (torch.Tensor): features to be flattened

        Returns:
            torch.Tensor: flattened feature vectors of shape (batch, -1)
        """
        return features.reshape(len(features), -1)

    def _all_contrastive_loss_modules_defined(self) -> bool:
        """
        Checks whether all of the components required to compute the PerFCL features and loss function are defined.
        There are instances where some are defined but not others. For example, in the very first round of training
        The initial_global_module will have been defined before training starts, but the old_local_module and
        old_global_module components will not have been.

        Returns:
            bool: Indicates True if all of the modules are not None
        """
        return (
            self.old_local_module is not None
            and self.old_global_module is not None
            and self.initial_global_module is not None
        )

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
        # For PerFCL models, we required the input to simply be a torch.Tensor
        assert isinstance(input, torch.Tensor)
        preds, features = self.model(input)
        # In the first server round, these module will not have been set.
        if (
            self.old_local_module is not None
            and self.old_global_module is not None
            and self.initial_global_module is not None
        ):
            # Pass the input through the old feature extractors and the initial global model after aggregation and
            # flatten them
            features["old_local_features"] = self._flatten(self.old_local_module.forward(input))
            features["old_global_features"] = self._flatten(self.old_global_module.forward(input))
            features["initial_global_features"] = self._flatten(self.initial_global_module.forward(input))

        return preds, features

    def update_after_train(self, local_steps: int, loss_dict: Dict[str, float], config: Config) -> None:
        """
        This function is called after client-side training concludes. In this case, it is used to save the local
        and global feature extraction weights/modules to be used in the next round of client-side training.

        Args:
            local_steps (int): Number of steps performed during training
            loss_dict (Dict[str, float]): Losses computed during training.
            config (Config): The config from the server
        """
        assert isinstance(self.model, PerFclModel)
        # First module is the local feature extractor for PerFcl Models
        self.old_local_module = self.clone_and_freeze_model(self.model.first_feature_extractor)
        # Second module is the global feature extractor for PerFcl Models
        self.old_global_module = self.clone_and_freeze_model(self.model.second_feature_extractor)

        super().update_after_train(local_steps, loss_dict, config)

    def update_before_train(self, current_server_round: int) -> None:
        """
        This function is called prior to the start of client-side training, but after the server parameters have be
        received and injected into the model. In this case, it is used to save the aggregated global feature extractor
        weights/module representing the initial state of this module BEFORE this iteration of client-side training
        but AFTER server-side aggregation.

        Args:
            current_server_round (int): Current server round being performed.
        """
        # Save the parameters of the aggregated global model
        assert isinstance(self.model, PerFclModel)
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
        For PerFCL, the total loss is the standard criterion loss provided by the user and the PerFCL contrastive
        losses aimed at manipulating the local and global feature extractor latent spaces.

        Args:
            preds (Dict[str, torch.Tensor]): Prediction(s) of the model(s) indexed by name.
            features (Dict[str, torch.Tensor]): Feature(s) of the model(s) indexed by name.
            target (torch.Tensor): Ground truth data to evaluate predictions against.

        Returns:
            Tuple[torch.Tensor, Union[Dict[str, torch.Tensor], None]]; A tuple with:
                - The tensor for the total loss
                - A dictionary with `loss`, `total_loss`, `global_feature_contrastive_loss`, and
                    `local_feature_contrastive_loss` representing the various and relevant pieces of the loss
                    calculations
        """

        loss = self.criterion(preds["prediction"], target)
        # If any of these are None then we don't compute the PerFCL loss. This will happen on the first client-side
        # training run.
        if self.old_local_module is None or self.old_global_module is None or self.initial_global_module is None:
            return loss, {"loss": loss}

        total_loss = loss.clone()
        global_feature_contrastive_loss, local_feature_contrastive_loss = self.perfcl_loss_function(
            features["local_features"],
            features["old_local_features"],
            features["global_features"],
            features["old_global_features"],
            features["initial_global_features"],
        )

        total_loss += (
            self.global_feature_contrastive_loss_weight * global_feature_contrastive_loss
            + self.local_feature_contrastive_loss_weight * local_feature_contrastive_loss
        )

        additional_losses = {
            "loss": loss,
            "global_feature_contrastive_loss": global_feature_contrastive_loss,
            "local_feature_contrastive_loss": local_feature_contrastive_loss,
            "total_loss": total_loss,
        }

        return total_loss, additional_losses

    def compute_evaluation_loss(
        self,
        preds: TorchPredType,
        features: TorchFeatureType,
        target: TorchTargetType,
    ) -> EvaluationLosses:
        """
        Computes evaluation loss given predictions of the model and ground truth data. Also computes
        additional loss components associated with the PerFCL loss function.

        Args:
            preds (Dict[str, torch.Tensor]): Prediction(s) of the model(s) indexed by name.
                All predictions included in dictionary will be used to compute metrics.
            features: (Dict[str, torch.Tensor]): Feature(s) of the model(s) indexed by name.
            target: (torch.Tensor): Ground truth data to evaluate predictions against.

        Returns:
            EvaluationLosses: an instance of EvaluationLosses containing checkpoint loss and additional losses
                indexed by name.
        """
        _, additional_losses = self.compute_loss_and_additional_losses(preds, features, target)
        return EvaluationLosses(checkpoint=additional_losses["loss"], additional_losses=additional_losses)
