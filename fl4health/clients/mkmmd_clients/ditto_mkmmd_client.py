from logging import ERROR, INFO
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from flwr.common.logger import log
from flwr.common.typing import Config, Scalar

from fl4health.checkpointing.client_module import CheckpointMode, ClientCheckpointModule
from fl4health.clients.ditto_client import DittoClient
from fl4health.losses.mkmmd_loss import MkMmdLoss
from fl4health.model_bases.feature_extractor_buffer import FeatureExtractorBuffer
from fl4health.utils.losses import LossMeterType
from fl4health.utils.metrics import Metric
from fl4health.utils.typing import TorchFeatureType, TorchInputType, TorchPredType, TorchTargetType


class DittoMkMmdClient(DittoClient):
    def __init__(
        self,
        data_path: Path,
        metrics: Sequence[Metric],
        device: torch.device,
        loss_meter_type: LossMeterType = LossMeterType.AVERAGE,
        checkpointer: Optional[ClientCheckpointModule] = None,
        lam: float = 1.0,
        mkmmd_loss_weight: float = 10.0,
        flatten_feature_extraction_layers: Optional[Dict[str, bool]] = None,
        feature_l2_norm_weight: float = 0.0,
        beta_global_update_interval: int = 20,
    ) -> None:
        """
        This client implements the MK-MMD loss function in the Ditto framework. The MK-MMD loss is a measure of the
        distance between the distributions of the features of the local model and initial global model of each round.
        The MK-MMD loss is added to the local loss to penalize the local model for drifting away from the global model.

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
            lam (float, optional): weight applied to the Ditto drift loss. Defaults to 1.0.
            mkmmd_loss_weight (float, optional): weight applied to the MK-MMD loss. Defaults to 10.0.
            flatten_feature_extraction_layers (Optional[Dict[str, bool]], optional): Dictionary of layers to extract
                features from them and whether to flatten them. Keys are the layer names that are extracted from the
                named_modules and values are boolean. Defaults to None.
            feature_l2_norm_weight (float, optional): weight applied to the L2 norm of the features.
                Defaults to 0.0.
            beta_global_update_interval (int, optional): interval at which to update the betas for the MK-MMD loss. If
                set to above 0, the betas will be updated based on whole distribution of latent features of data with
                the given update interval. If set to 0, the betas will not be updated. If set to -1, the betas will be
                updated after each individual batch based on only that individual batch. Defaults to 20.
        """
        super().__init__(
            data_path=data_path,
            metrics=metrics,
            device=device,
            loss_meter_type=loss_meter_type,
            checkpointer=checkpointer,
            lam=lam,
        )
        self.mkmmd_loss_weight = mkmmd_loss_weight
        if self.mkmmd_loss_weight == 0:
            log(
                ERROR,
                "MK-MMD loss weight is set to 0. As MK-MMD loss will not be computed, ",
                "please use vanilla DittoClient instead.",
            )

        self.feature_l2_norm_weight = feature_l2_norm_weight
        self.beta_global_update_interval = beta_global_update_interval
        if self.beta_global_update_interval == -1:
            log(INFO, "Betas for the MK-MMD loss will be updated for each individual batch.")
        elif self.beta_global_update_interval == 0:
            log(INFO, "Betas for the MK-MMD loss will not be updated.")
        elif self.beta_global_update_interval > 0:
            log(INFO, f"Betas for the MK-MMD loss will be updated every {self.beta_global_update_interval} steps.")
        else:
            raise ValueError("Invalid beta_global_update_interval. It should be either -1, 0 or a positive integer.")
        if flatten_feature_extraction_layers:
            self.flatten_feature_extraction_layers = flatten_feature_extraction_layers
        else:
            self.flatten_feature_extraction_layers = {}
        self.mkmmd_losses = {}
        for layer in self.flatten_feature_extraction_layers.keys():
            self.mkmmd_losses[layer] = MkMmdLoss(
                device=self.device, minimize_type_two_error=True, normalize_features=True, layer_name=layer
            ).to(self.device)

        self.initial_global_model: nn.Module
        self.local_feature_extractor: FeatureExtractorBuffer
        self.initial_global_feature_extractor: FeatureExtractorBuffer

    def setup_client(self, config: Config) -> None:
        super().setup_client(config)
        self.local_feature_extractor = FeatureExtractorBuffer(
            model=self.model,
            flatten_feature_extraction_layers=self.flatten_feature_extraction_layers,
        )

    def update_before_train(self, current_server_round: int) -> None:
        super().update_before_train(current_server_round)
        assert isinstance(self.global_model, nn.Module)
        # Register hooks to extract features from the local model if not already registered
        self.local_feature_extractor._maybe_register_hooks()
        # Clone and freeze the initial weights GLOBAL MODEL. These are used to form the Ditto local
        # update penalty term.
        self.initial_global_model = self.clone_and_freeze_model(self.global_model)
        self.initial_global_feature_extractor = FeatureExtractorBuffer(
            model=self.initial_global_model,
            flatten_feature_extraction_layers=self.flatten_feature_extraction_layers,
        )
        # Register hooks to extract features from the initial global model if not already registered
        self.initial_global_feature_extractor._maybe_register_hooks()

    def _should_optimize_betas(self, step: int) -> bool:
        step_at_interval = (step - 1) % self.beta_global_update_interval == 0
        valid_components_present = self.initial_global_model is not None
        return step_at_interval and valid_components_present

    def update_after_step(self, step: int, current_round: Optional[int] = None) -> None:
        if self.beta_global_update_interval > 0 and self._should_optimize_betas(step):
            # If the mkmmd loss doesn't matter, we don't bother optimizing betas
            if self.mkmmd_loss_weight != 0:
                return super().update_after_step(step)

            # Get the feature distribution of the local and initial global features with evaluation
            # mode
            local_distributions, initial_global_distributions = self.update_buffers(
                self.model, self.initial_global_model
            )
            # Update betas for the MK-MMD loss based on gathered features during training
            for layer, layer_mkmmd_loss in self.mkmmd_losses.items():
                layer_mkmmd_loss.betas = layer_mkmmd_loss.optimize_betas(
                    X=local_distributions[layer], Y=initial_global_distributions[layer], lambda_m=1e-5
                )

        return super().update_after_step(step)

    def update_buffers(
        self, local_model: torch.nn.Module, initial_global_model: torch.nn.Module
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Update the feature buffer of the local and global features."""

        self.local_feature_extractor.clear_buffers()
        self.initial_global_feature_extractor.clear_buffers()

        self.local_feature_extractor.enable_accumulating_features()
        self.initial_global_feature_extractor.enable_accumulating_features()

        # Save the initial state of the local model to restore it after the buffer is populated,
        # however as initial global model is already cloned and frozen, we don't need to save its state.
        initial_state_local_model = local_model.training

        # Set local model to evaluation mode, as we don't want to create a computational graph
        # for the local model when populating the local buffer with features to compute optimal
        # betas for the MK-MMD loss
        local_model.eval()

        # Make sure the local model is in evaluation mode before populating the local buffer
        assert not local_model.training

        # Make sure the initial global model is in evaluation mode before populating the global buffer
        # as it is already cloned and frozen from the global model
        assert not initial_global_model.training

        with torch.no_grad():
            for input, _ in self.train_loader:
                input = input.to(self.device)
                # Pass the input through the local model to populate the local_feature_extractor buffer
                local_model(input)
                # Pass the input through the initial global model to populate the local_feature_extractor buffer
                initial_global_model(input)
        local_distributions = self.local_feature_extractor.get_extracted_features()
        initial_global_distributions = self.initial_global_feature_extractor.get_extracted_features()
        # Restore the initial state of the local model
        if initial_state_local_model:
            local_model.train()

        self.local_feature_extractor.disable_accumulating_features()
        self.initial_global_feature_extractor.disable_accumulating_features()

        self.local_feature_extractor.clear_buffers()
        self.initial_global_feature_extractor.clear_buffers()

        return local_distributions, initial_global_distributions

    def predict(
        self,
        input: TorchInputType,
    ) -> Tuple[TorchPredType, TorchFeatureType]:
        """
         Computes the predictions for both the GLOBAL and LOCAL models and pack them into the prediction dictionary

         Args:
             input (TorchInputType): Inputs to be fed into the model. If input is
                 of type Dict[str, torch.Tensor], it is assumed that the keys of
                 input match the names of the keyword arguments of self.model.
                 forward().

         Returns:
             Tuple[TorchPredType, TorchFeatureType]: A tuple in which the
                 first element contains a dictionary of predictions indexed by
                 name and the second element contains intermediate activations
                 indexed by name. By passing features, we can compute losses
                 such as the model contrasting loss in MOON. All predictions
                 included in dictionary will by default be used to compute
                 metrics seperately.

        Raises:
             TypeError: Occurs when something other than a tensor or dict of tensors is passed in to the model's
             forward method.
             ValueError: Occurs when something other than a tensor or dict of tensors is returned by the model
             forward.
        """

        # We use features from initial_global_model to compute the MK-MMD loss not the global_model
        global_preds = self.global_model(input)
        local_preds = self.model(input)
        features = self.local_feature_extractor.get_extracted_features()
        if self.mkmmd_loss_weight != 0:
            # Compute the features of the initial_global_model
            self.initial_global_model(input)
            initial_global_features = self.initial_global_feature_extractor.get_extracted_features()
            for key in initial_global_features.keys():
                features[" ".join(["init_global", key])] = initial_global_features[key]

        return {"global": global_preds, "local": local_preds}, features

    def _maybe_checkpoint(self, loss: float, metrics: Dict[str, Scalar], checkpoint_mode: CheckpointMode) -> None:
        # Hooks need to be removed before checkpointing the model
        self.local_feature_extractor.remove_hooks()
        super()._maybe_checkpoint(loss=loss, metrics=metrics, checkpoint_mode=checkpoint_mode)

    def compute_loss_and_additional_losses(
        self, preds: TorchPredType, features: TorchFeatureType, target: TorchTargetType
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Computes the loss and any additional losses given predictions of the model and ground truth data.

        Args:
            preds (TorchPredType): Prediction(s) of the model(s) indexed by name.
            features (TorchFeatureType): Feature(s) of the model(s) indexed by name.
            target (TorchTargetType): Ground truth data to evaluate predictions against.

        Returns:
            Tuple[torch.Tensor, Union[Dict[str, torch.Tensor], None]]; A tuple with:
                - The tensor for the loss
                - A dictionary of additional losses with their names and values, or None if
                    there are no additional losses.
        """
        total_loss, additional_losses = super().compute_loss_and_additional_losses(preds, features, target)

        if self.mkmmd_loss_weight != 0:
            if self.beta_global_update_interval == -1:
                # Update betas for the MK-MMD loss based on computed features during training
                for layer, layer_mkmmd_loss in self.mkmmd_losses.items():
                    layer_mkmmd_loss.betas = layer_mkmmd_loss.optimize_betas(
                        X=features[layer], Y=features[" ".join(["init_global", layer])], lambda_m=1e-5
                    )
            # Compute MK-MMD loss
            total_mkmmd_loss = torch.tensor(0.0, device=self.device)
            for layer, layer_mkmmd_loss in self.mkmmd_losses.items():
                mkmmd_loss = layer_mkmmd_loss(features[layer], features[" ".join(["init_global", layer])])
                additional_losses["_".join(["mkmmd_loss", layer])] = mkmmd_loss
                total_mkmmd_loss += mkmmd_loss
            total_loss += self.mkmmd_loss_weight * total_mkmmd_loss
            additional_losses["mkmmd_loss_total"] = total_mkmmd_loss
        if self.feature_l2_norm_weight:
            # Compute the average L2 norm of the features over the batch
            feature_l2_norm_loss = torch.linalg.norm(features["features"]) / len(features["features"])
            total_loss += self.feature_l2_norm_weight * feature_l2_norm_loss
            additional_losses["feature_l2_norm_loss"] = feature_l2_norm_loss

        additional_losses["total_loss"] = total_loss

        return total_loss, additional_losses