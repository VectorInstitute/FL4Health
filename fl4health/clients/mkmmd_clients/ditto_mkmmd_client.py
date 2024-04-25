from logging import ERROR, INFO
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from flwr.common.logger import log
from flwr.common.typing import Config

from fl4health.checkpointing.checkpointer import TorchCheckpointer
from fl4health.clients.basic_client import TorchInputType
from fl4health.clients.ditto_client import DittoClient
from fl4health.losses.mkmmd_loss import MkMmdLoss
from fl4health.model_bases.feature_extractor_buffer import FeatureExtractorBuffer
from fl4health.utils.losses import LossMeterType
from fl4health.utils.metrics import Metric


class DittoMkmmdClient(DittoClient):
    def __init__(
        self,
        data_path: Path,
        metrics: Sequence[Metric],
        device: torch.device,
        loss_meter_type: LossMeterType = LossMeterType.AVERAGE,
        checkpointer: Optional[TorchCheckpointer] = None,
        lam: float = 1.0,
        mkmmd_loss_weight: float = 10.0,
        flatten_feature_extraction_layers: Dict[str, bool] = {},
        feature_l2_norm_weight: float = 0.0,
        beta_global_update_interval: int = 20,
    ) -> None:
        """
        This client implements the MK-MMD loss function in the Ditto framework. The MK-MMD loss is a measure of the
        distance between the distributions of the features of the local model and init global of each round. The MK-MMD
        loss is added to the local loss to penalize the local model for drifting away from the global model.

        Args:
            data_path (Path): path to the data to be used to load the data for client-side training
            metrics (Sequence[Metric]): Metrics to be computed based on the labels and predictions of the client model
            device (torch.device): Device indicator for where to send the model, batches, labels etc. Often 'cpu' or
                'cuda'
            loss_meter_type (LossMeterType, optional): Type of meter used to track and compute the losses over
                each batch. Defaults to LossMeterType.AVERAGE.
            checkpointer (Optional[TorchCheckpointer], optional): Checkpointer to be used for client-side
                checkpointing. Defaults to None.
            metrics_reporter (Optional[MetricsReporter], optional): A metrics reporter instance to record the metrics
                during the execution. Defaults to an instance of MetricsReporter with default init parameters.
            lam (float, optional): weight applied to the Ditto drift loss. Defaults to 1.0.
            mkmmd_loss_weight (float, optional): weight applied to the MK-MMD loss. Defaults to 10.0.
            flatten_feature_extraction_layers (Dict[str, bool], optional): Dictionary of layers to extract features
                from them and whether to flatten them. Defaults to {}.
            feature_l2_norm_weight (float, optional): weight applied to the L2 norm of the features.
                Defaults to 0.0.
            beta_global_update_interval (int, optional): interval at which to update the betas for the
                MK-MMD loss. Defaults to 20. If set to -1, the betas will be updated for each individual batch.
                If set to 0, the betas will not be updated.
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
        self.flatten_feature_extraction_layers = flatten_feature_extraction_layers
        self.mkmmd_losses = {}
        for layer in self.flatten_feature_extraction_layers.keys():
            self.mkmmd_losses[layer] = MkMmdLoss(device=self.device, minimize_type_two_error=True).to(self.device)

        self.init_global_model: nn.Module
        self.local_feature_extractor: FeatureExtractorBuffer
        self.init_global_feature_extractor: FeatureExtractorBuffer

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
        self.init_global_model = self.clone_and_freeze_model(self.global_model)
        self.init_global_feature_extractor = FeatureExtractorBuffer(
            model=self.init_global_model,
            flatten_feature_extraction_layers=self.flatten_feature_extraction_layers,
        )
        # Register hooks to extract features from the init global model if not already registered
        self.init_global_feature_extractor._maybe_register_hooks()

    def _should_optimize_betas(self, step: int) -> bool:
        assert self.beta_global_update_interval is not None
        step_at_interval = (step - 1) % self.beta_global_update_interval == 0
        valid_components_present = self.init_global_model is not None
        return step_at_interval and valid_components_present

    def update_after_step(self, step: int) -> None:
        if self.beta_global_update_interval is not None and self._should_optimize_betas(step):
            assert self.init_global_model is not None
            # Get the feature distribution of the local and init global features with evaluation mode
            local_distributions, init_global_distributions = self.update_buffers(self.model, self.init_global_model)
            # Update betas for the MK-MMD loss based on gathered features during training
            if self.mkmmd_loss_weight != 0:
                for layer in self.flatten_feature_extraction_layers.keys():
                    self.mkmmd_losses[layer].betas = self.mkmmd_losses[layer].optimize_betas(
                        X=local_distributions[layer], Y=init_global_distributions[layer], lambda_m=1e-5
                    )

        return super().update_after_step(step)

    def update_buffers(
        self, local_model: torch.nn.Module, init_global_model: torch.nn.Module
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Update the feature buffer of the local and global features."""

        self.local_feature_extractor.clear_buffers()
        self.init_global_feature_extractor.clear_buffers()

        self.local_feature_extractor.enable_accumulating_features()
        self.init_global_feature_extractor.enable_accumulating_features()

        # Save the initial state of the local model to restore it after the buffer is populated,
        # however as init global model is already cloned and frozen, we don't need to save its state.
        init_state_local_model = local_model.training

        # Set local model to evaluation mode, as we don't want to create a computational graph
        # for the local model when populating the local buffer with features to compute optimal
        # betas for the MK-MMD loss
        local_model.eval()

        # Make sure the local model is in evaluation mode before populating the local buffer
        assert not local_model.training

        # Make sure the init global model is in evaluation mode before populating the global buffer
        # as it is already cloned and frozen from the global model
        assert not init_global_model.training

        with torch.no_grad():
            for i, (input, _) in enumerate(self.train_loader):
                input = input.to(self.device)
                # Pass the input through the local model to populate the local_feature_extractor buffer
                _ = local_model(input)
                # Pass the input through the init global model to populate the local_feature_extractor buffer
                _ = init_global_model(input)
        local_distributions: Dict[str, torch.Tensor] = self.local_feature_extractor.get_extracted_features()
        init_global_distributions: Dict[
            str, torch.Tensor
        ] = self.init_global_feature_extractor.get_extracted_features()
        # Restore the initial state of the local model
        if init_state_local_model:
            local_model.train()

        self.local_feature_extractor.disable_accumulating_features()
        self.init_global_feature_extractor.disable_accumulating_features()

        self.local_feature_extractor.clear_buffers()
        self.init_global_feature_extractor.clear_buffers()

        return local_distributions, init_global_distributions

    def predict(
        self,
        input: TorchInputType,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Computes the predictions for both the GLOBAL and LOCAL models and pack them into the prediction dictionary

        Args:
            input (Union[torch.Tensor, Dict[str, torch.Tensor]]): Inputs to be fed into both models.

        Returns:
            Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]: A tuple in which the first element
            contains predictions indexed by name and the second element contains intermediate activations
            index by name.

        Raises:
            ValueError: Occurs when something other than a tensor or dict of tensors is returned by the model
            forward.
        """

        # We use features from init_global_model to compute the MK-MMD loss not the global_model
        global_preds = self.global_model(input)
        local_preds = self.model(input)
        features = self.local_feature_extractor.get_extracted_features()
        if self.mkmmd_loss_weight != 0:
            # Compute the features of the init_global_model
            _ = self.init_global_model(input)
            init_global_features = self.init_global_feature_extractor.get_extracted_features()
            for key in init_global_features.keys():
                features[" ".join(["init_global", key])] = init_global_features[key]

        return {"global": global_preds, "local": local_preds}, features

    def _maybe_checkpoint(self, current_metric_value: float) -> None:
        # Hooks need to be removed before checkpointing the model
        self.local_feature_extractor.remove_hooks()
        super()._maybe_checkpoint(current_metric_value)

    def compute_loss_and_additional_losses(
        self,
        preds: Dict[str, torch.Tensor],
        features: Dict[str, torch.Tensor],
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Computes the loss and any additional losses given predictions of the model and ground truth data.

        Args:
            preds (Dict[str, torch.Tensor]): Prediction(s) of the model(s) indexed by name.
            features (Dict[str, torch.Tensor]): Feature(s) of the model(s) indexed by name.
            target (torch.Tensor): Ground truth data to evaluate predictions against.

        Returns:
            Tuple[torch.Tensor, Dict[str, torch.Tensor]]; A tuple with:
                - The tensor for the total loss
                - A dictionary with `local_loss`, `global_loss`, `total_loss` and, based on client attributes set
                from server config, also `mkmmd_loss`, `feature_l2_norm_loss` keys and their respective calculated
                values.
        """
        total_loss, additional_losses = super().compute_loss_and_additional_losses(preds, features, target)

        if self.mkmmd_loss_weight != 0:
            if self.beta_global_update_interval is None:
                # Update betas for the MK-MMD loss based on computed features during training
                for layer in self.flatten_feature_extraction_layers.keys():
                    self.mkmmd_losses[layer].betas = self.mkmmd_losses[layer].optimize_betas(
                        X=features[layer], Y=features[" ".join(["init_global", layer])], lambda_m=1e-5
                    )
            # Compute MK-MMD loss
            mkmmd_loss = torch.tensor(0.0, device=self.device)
            for layer in self.flatten_feature_extraction_layers.keys():
                mkmmd_loss += self.mkmmd_losses[layer](features[layer], features[" ".join(["init_global", layer])])
            total_loss += self.mkmmd_loss_weight * mkmmd_loss
            additional_losses["mkmmd_loss"] = mkmmd_loss
        if self.feature_l2_norm_weight:
            # Compute the average L2 norm of the features over the batch
            feature_l2_norm_loss = torch.linalg.norm(features["features"]) / len(features["features"])
            total_loss += self.feature_l2_norm_weight * feature_l2_norm_loss
            additional_losses["feature_l2_norm_loss"] = feature_l2_norm_loss

        additional_losses["total_loss"] = total_loss

        return total_loss, additional_losses
