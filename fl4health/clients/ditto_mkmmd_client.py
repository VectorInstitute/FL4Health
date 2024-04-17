from logging import INFO
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
        beta_update_interval: int = 20,
        feature_l2_norm_weight: Optional[float] = 0.0,
    ) -> None:
        """
        This client implements the Ditto algorithm from Ditto: Fair and Robust Federated Learning Through
        Personalization. The idea is that we want to train personalized versions of the global model for each client.
        So we simultaneously train a global model that is aggregated on the server-side and use those weights to also
        constrain the training of a local model. The constraint for this local model is identical to the FedProx loss.

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
            beta_update_interval (int, optional): interval at which to update the betas for the MK-MMD loss.
                Defaults to 20.
            feature_l2_norm_weight (Optional[float], optional): weight applied to the L2 norm of the features.
            Defaults to 0.0.
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
        self.feature_l2_norm_weight = feature_l2_norm_weight
        self.beta_update_interval = beta_update_interval
        self.mkmmd_loss = MkMmdLoss(device=self.device, minimize_type_two_error=True).to(self.device)

        self.init_global_model: nn.Module

    def update_before_train(self, current_server_round: int) -> None:
        assert isinstance(self.global_model, nn.Module)
        # Clone and freeze the initial weights GLOBAL MODEL. These are used to form the Ditto local
        # update penalty term.
        self.init_global_model = self.clone_and_freeze_model(self.global_model)

        return super().update_before_train(current_server_round)

    def update_after_step(self, step: int) -> None:
        if step % self.beta_update_interval == 0:
            if self.mkmmd_loss_weight and self.init_global_model:
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

        init_state_local_model = local_model.training

        # Set local model to evaluation mode
        local_model.eval()

        assert not local_model.training
        assert not init_global_model.training

        local_buffer = []
        init_global_buffer = []

        with torch.no_grad():
            for input, target in self.train_loader:
                input, target = input.to(self.device), target.to(self.device)
                _, local_features = local_model(input)
                _, init_global_features = init_global_model(input)

                local_buffer.append(local_features["features"].reshape(len(local_features["features"]), -1))
                init_global_buffer.append(
                    init_global_features["features"].reshape(len(init_global_features["features"]), -1)
                )

        if init_state_local_model:
            local_model.train()

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

        global_preds, _ = self.global_model(input)
        local_preds, features = self.model(input)

        if self.mkmmd_loss_weight != 0:
            if not isinstance(self.model, MoonModel) or not isinstance(self.init_global_model, MoonModel):
                AssertionError(
                    "To compute the MK-MMD loss, the client model and the init_global_model must be of type MoonModel."
                )
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
            Tuple[torch.Tensor, Union[Dict[str, torch.Tensor], None]]; A tuple with:
                - The tensor for the total loss
                - A dictionary with `local_loss`, `global_loss`, `total_loss` and, based on client attributes set
                from server config, also `mkmmd_loss`, `feature_l2_norm_loss` keys and their respective calculated
                values.
        """
        total_loss, additional_losses = super().compute_loss_and_additional_losses(preds, features, target)

        if self.mkmmd_loss_weight != 0:
            assert "init_global_features" in features
            assert "features" in features
            # Compute MK-MMD loss
            mkmmd_loss = self.mkmmd_loss(features["features"], features["init_global_features"])
            total_loss += self.mkmmd_loss_weight * mkmmd_loss
            additional_losses["mkmmd_loss"] = mkmmd_loss
            if self.feature_l2_norm_weight:
                feature_l2_norm_loss = torch.linalg.norm(features["features"])
                total_loss += self.feature_l2_norm_weight * feature_l2_norm_loss
                additional_losses["feature_l2_norm_loss"] = feature_l2_norm_loss

        additional_losses["total_loss"] = total_loss

        return total_loss, additional_losses
