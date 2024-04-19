from logging import ERROR, INFO
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from flwr.common.logger import log

from fl4health.checkpointing.checkpointer import TorchCheckpointer
from fl4health.clients.basic_client import TorchInputType
from fl4health.clients.ditto_client import DittoClient
from fl4health.losses.mkmmd_loss import MkMmdLoss
from fl4health.model_bases.moon_base import MoonModel
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
        feature_l2_norm_weight: float = 0.0,
        beta_global_update_interval: Optional[int] = 20,
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
            feature_l2_norm_weight (float, optional): weight applied to the L2 norm of the features.
            Defaults to 0.0.
            beta_global_update_interval (Optional[int], optional): interval at which to update the betas for the
                MK-MMD loss. Defaults to 20. If set to None, the betas will be updated for each individual batch.
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
        if self.beta_global_update_interval is None:
            log(INFO, "Betas for the MK-MMD loss will be updated for each individual batch.")
        elif self.beta_global_update_interval == 0:
            log(INFO, "Betas for the MK-MMD loss will not be updated.")
        else:
            log(INFO, f"Betas for the MK-MMD loss will be updated every {self.beta_global_update_interval} steps.")

        self.mkmmd_loss = MkMmdLoss(device=self.device, minimize_type_two_error=True).to(self.device)

        self.init_global_model: nn.Module

    def update_before_train(self, current_server_round: int) -> None:
        assert isinstance(self.global_model, nn.Module)
        # Clone and freeze the initial weights GLOBAL MODEL. These are used to form the Ditto local
        # update penalty term.
        self.init_global_model = self.clone_and_freeze_model(self.global_model)

        return super().update_before_train(current_server_round)

    def _should_optimize_betas(self, step: int) -> bool:
        assert self.beta_global_update_interval is not None
        step_at_interval = step % self.beta_global_update_interval == 0
        valid_components_present = self.init_global_model is not None
        return step_at_interval and valid_components_present

    def update_after_step(self, step: int) -> None:
        if self.beta_global_update_interval is not None and self._should_optimize_betas(step):
            assert self.init_global_model is not None
            # Get the feature distribution of the local and init global features with evaluation mode
            local_distribution, init_global_distribution = self.update_buffers(self.model, self.init_global_model)
            # Update betas for the MK-MMD loss based on gathered features during training
            if self.mkmmd_loss_weight != 0:
                self.mkmmd_loss.betas = self.mkmmd_loss.optimize_betas(
                    X=local_distribution, Y=init_global_distribution, lambda_m=1e-5
                )
                log(INFO, f"Set optimized betas to minimize distance: {self.mkmmd_loss.betas.squeeze()}.")

        return super().update_after_step(step)

    def update_buffers(
        self, local_model: torch.nn.Module, init_global_model: torch.nn.Module
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update the feature buffer of the local and global features."""
        assert isinstance(local_model, MoonModel)
        assert isinstance(init_global_model, MoonModel)

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

        local_buffer = []
        init_global_buffer = []

        with torch.no_grad():
            for input, _ in self.train_loader:
                input = input.to(self.device)
                _, local_features = local_model(input)
                _, init_global_features = init_global_model(input)

                # Flatten the features to compute optimal betas for the MK-MMD loss
                local_buffer.append(local_features["features"].reshape(len(local_features["features"]), -1))
                init_global_buffer.append(
                    init_global_features["features"].reshape(len(init_global_features["features"]), -1)
                )

        # Restore the initial state of the local model
        if init_state_local_model:
            local_model.train()

        # The buffers are in shape (batch_size, feature_size). We tack them along the batch dimension
        # (dim=0) to get a tensor of shape (num_samples, feature_size)
        return torch.cat(local_buffer, dim=0), torch.cat(init_global_buffer, dim=0)

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
        assert isinstance(self.model, MoonModel)
        assert isinstance(self.global_model, MoonModel)

        # We use features from init_global_model to compute the MK-MMD loss not the global_model
        global_preds, _ = self.global_model(input)
        local_preds, features = self.model(input)

        if self.mkmmd_loss_weight != 0:
            if not isinstance(self.model, MoonModel) or not isinstance(self.init_global_model, MoonModel):
                raise AssertionError(
                    "To compute the MK-MMD loss, the client model and the init_global_model must be of type MoonModel."
                )
            # Compute the features of the init_global_model
            _, init_global_features = self.init_global_model(input)
            features.update({"init_global_features": init_global_features["features"]})

        return {"global": global_preds["prediction"], "local": local_preds["prediction"]}, features

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
            assert "init_global_features" in features
            assert "features" in features
            if self.beta_global_update_interval is None:
                # Update betas for the MK-MMD loss based on computed features during training
                self.mkmmd_loss.betas = self.mkmmd_loss.optimize_betas(
                    X=features["features"], Y=features["init_global_features"], lambda_m=1e-5
                )
            # Compute MK-MMD loss
            mkmmd_loss = self.mkmmd_loss(features["features"], features["init_global_features"])
            total_loss += self.mkmmd_loss_weight * mkmmd_loss
            additional_losses["mkmmd_loss"] = mkmmd_loss
        if self.feature_l2_norm_weight:
            # Compute the average L2 norm of the features over the batch
            feature_l2_norm_loss = torch.linalg.norm(features["features"]) / len(features["features"])
            total_loss += self.feature_l2_norm_weight * feature_l2_norm_loss
            additional_losses["feature_l2_norm_loss"] = feature_l2_norm_loss

        additional_losses["total_loss"] = total_loss

        return total_loss, additional_losses
