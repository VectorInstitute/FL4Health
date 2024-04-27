from logging import ERROR
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import torch
from flwr.common.logger import log

from fl4health.checkpointing.checkpointer import TorchCheckpointer
from fl4health.clients.basic_client import TorchInputType
from fl4health.clients.fenda_client import FendaClient
from fl4health.losses.mkmmd_loss import MkMmdLoss
from fl4health.model_bases.fenda_base import FendaModel
from fl4health.utils.losses import LossMeterType
from fl4health.utils.metrics import Metric


class FendaMkmmdClient(FendaClient):
    def __init__(
        self,
        data_path: Path,
        metrics: Sequence[Metric],
        device: torch.device,
        loss_meter_type: LossMeterType = LossMeterType.AVERAGE,
        checkpointer: Optional[TorchCheckpointer] = None,
        temperature: float = 0.5,
        perfcl_loss_weights: Optional[Tuple[float, float]] = None,
        cos_sim_loss_weight: Optional[float] = None,
        contrastive_loss_weight: Optional[float] = None,
        mkmmd_loss_weights: Optional[Tuple[float, float]] = None,
    ) -> None:
        super().__init__(
            data_path=data_path,
            metrics=metrics,
            device=device,
            loss_meter_type=loss_meter_type,
            checkpointer=checkpointer,
            temperature=temperature,
            perfcl_loss_weights=perfcl_loss_weights,
            cos_sim_loss_weight=cos_sim_loss_weight,
            contrastive_loss_weight=contrastive_loss_weight,
        )
        """
        This module is used to implement the FENDA client with MK-MMD loss. The MK-MMD loss is used to minimize the
        distance between the global and aggregated global features and maximize the distance between the local and
        aggregated global features.

        Args:
            data_path: Path to the data directory.
            metrics: List of metrics to be used for evaluation.
            device: Device to be used for training.
            loss_meter_type: Type of loss meter to be used.
            checkpointer: Checkpointer to be used for checkpointing.
            temperature: Temperature to be used for contrastive loss.
            perfcl_loss_weights: Weights to be used for perfcl loss.
            Each value associate with one of two contrastive losses in perfcl loss.
            cos_sim_loss_weight: Weight to be used for cosine similarity loss.
            contrastive_loss_weight: Weight to be used for contrastive loss.
            mkmmd_loss_weights: Weights to be used for mkmmd losses. First value is for minimizing the distance
            and second value is for maximizing the distance.
        """

        self.mkmmd_loss_weights = mkmmd_loss_weights
        if self.mkmmd_loss_weights == (0, 0):
            log(
                ERROR,
                "MK-MMD loss weight is set to (0,0). As none of MK-MMD losses will not be computed, ",
                "please use vanilla FendaClient instead.",
            )
        self.mkmmd_loss_min = MkMmdLoss(device=self.device, minimize_type_two_error=True, normalize_features=True).to(
            self.device
        )
        self.mkmmd_loss_max = MkMmdLoss(device=self.device, minimize_type_two_error=False, normalize_features=True).to(
            self.device
        )

    def predict(self, input: TorchInputType) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Computes the prediction(s) and features of the model(s) given the input.

        Args:
            input (TorchInputType): Inputs to be fed into the model. TorchInputType is simply an alias
            for the union of torch.Tensor and Dict[str, torch.Tensor].

        Returns:
            Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]: A tuple in which the first element
            contains predictions indexed by name and the second element contains intermediate activations
            index by name. Specificaly the features of the model, features of the global model and features of
            the old model are returned. All predictions included in dictionary will be used to compute metrics.
        """
        preds, features = super().predict(input)
        if self.mkmmd_loss_weights is not None:
            if self.aggregated_global_module is not None:
                features["aggregated_global_features"] = self.aggregated_global_module.forward(input).reshape(
                    len(features["global_features"]), -1
                )

        return preds, features

    def update_after_train(self, local_steps: int, loss_dict: Dict[str, float]) -> None:
        # Save the parameters of the old model
        assert isinstance(self.model, FendaModel)
        if self.mkmmd_loss_weights:
            self.old_global_module = self.clone_and_freeze_model(self.model.global_module)

        return super().update_after_train(local_steps, loss_dict)

    def update_before_train(self, current_server_round: int) -> None:
        super().update_before_train(current_server_round)
        assert isinstance(self.model, FendaModel)
        if self.mkmmd_loss_weights and self.old_global_module and self.aggregated_global_module:
            local_distribution, old_global_distribution, aggregated_distribution = self.update_buffers(
                local_module=self.model.local_module,
                global_module=self.old_global_module,
                aggregated_module=self.aggregated_global_module,
            )
            if self.mkmmd_loss_weights[0] != 0.0:
                self.mkmmd_loss_min.betas = self.mkmmd_loss_min.optimize_betas(
                    X=old_global_distribution, Y=aggregated_distribution, lambda_m=1e-5
                )
            if self.mkmmd_loss_weights[1] != 0.0:
                self.mkmmd_loss_max.betas = self.mkmmd_loss_max.optimize_betas(
                    X=local_distribution, Y=aggregated_distribution, lambda_m=1e-5
                )

        return None

    def update_buffers(
        self, local_module: torch.nn.Module, global_module: torch.nn.Module, aggregated_module: torch.nn.Module
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Update the feature buffer of the local, global and aggregated features."""
        assert isinstance(local_module, torch.nn.Module)
        assert isinstance(global_module, torch.nn.Module)
        assert isinstance(aggregated_module, torch.nn.Module)

        local_buffer = []
        global_buffer = []
        aggregated_buffer = []

        # Save the state of the local and global modules, however as aggregated module is already
        # cloned and frozen, we don't need to save its state.
        init_state_local_module = local_module.training
        init_state_global_module = global_module.training

        # Compute the old features before aggregation and global features
        local_module.eval()
        global_module.eval()
        aggregated_module.eval()
        assert not local_module.training
        assert not global_module.training
        assert not aggregated_module.training

        with torch.no_grad():
            for input, target in self.train_loader:
                input, target = input.to(self.device), target.to(self.device)
                local_features = local_module.forward(input)
                global_features = global_module.forward(input)
                aggregated_features = aggregated_module.forward(input)

                # Local feature are same as old local features
                # Flatten the features to compute optimal betas for the MK-MMD loss
                local_buffer.append(local_features.reshape(len(local_features), -1))
                global_buffer.append(global_features.reshape(len(global_features), -1))
                aggregated_buffer.append(aggregated_features.reshape(len(aggregated_features), -1))

        # Reset the state of the local and global modules
        if init_state_local_module:
            local_module.train()
        if init_state_global_module:
            global_module.train()

        # The buffers are in shape (batch_size, feature_size). We tack them along the batch dimension
        # (dim=0) to get a tensor of shape (num_samples, feature_size)
        return torch.cat(local_buffer), torch.cat(global_buffer), torch.cat(aggregated_buffer)

    def get_mkmmd_loss(
        self,
        local_features: torch.Tensor,
        global_features: torch.Tensor,
        aggregated_global_features: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        MK-MMD loss aims to compute the distance among given two distributions with optimized betas.
        We compute the MK-MMD loss for minimizing the distance between the global and aggregated global features
        and maximizing the distance between the local and aggregated global features.
        """
        assert self.mkmmd_loss_weights is not None
        assert local_features.shape == aggregated_global_features.shape
        assert global_features.shape == aggregated_global_features.shape

        mkmmd_loss_min, mkmmd_loss_max = None, None
        if self.mkmmd_loss_weights[0] != 0.0:
            mkmmd_loss_min = self.mkmmd_loss_min(global_features, aggregated_global_features)
        if self.mkmmd_loss_weights[1] != 0.0:
            mkmmd_loss_max = self.mkmmd_loss_max(local_features, aggregated_global_features)
        return mkmmd_loss_min, mkmmd_loss_max

    def compute_loss_and_additional_losses(
        self,
        preds: Dict[str, torch.Tensor],
        features: Dict[str, torch.Tensor],
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Computes the loss and any additional losses given predictions of the model and ground truth data.
        In addition to inherited losses from parent FendaClient, this method also computes the MK-MMD losses
        if the weights are provided and adds them to the total loss and additional losses dictionary.
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

        total_loss, additional_losses = super().compute_loss_and_additional_losses(preds, features, target)

        if self.mkmmd_loss_weights and self.aggregated_global_module:
            mkmmd_loss_min, mkmmd_loss_max = self.get_mkmmd_loss(
                local_features=features["local_features"],
                global_features=features["global_features"],
                aggregated_global_features=features["aggregated_global_features"],
            )
            if mkmmd_loss_min:
                total_loss += self.mkmmd_loss_weights[0] * mkmmd_loss_min
                additional_losses["mkmmd_loss_min"] = mkmmd_loss_min

            if mkmmd_loss_max:
                total_loss -= self.mkmmd_loss_weights[1] * mkmmd_loss_max
                additional_losses["mkmmd_loss_max"] = mkmmd_loss_max

        additional_losses["total_loss"] = total_loss

        return total_loss, additional_losses
