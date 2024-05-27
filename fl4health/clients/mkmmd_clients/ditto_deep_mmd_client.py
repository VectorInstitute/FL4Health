from logging import ERROR
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from flwr.common.logger import log
from flwr.common.typing import Config, Scalar

from fl4health.checkpointing.client_module import CheckpointMode, ClientCheckpointModule
from fl4health.clients.basic_client import TorchInputType
from fl4health.clients.ditto_client import DittoClient
from fl4health.losses.deep_mmd_loss import DeepMmdLoss
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
        checkpointer: Optional[ClientCheckpointModule] = None,
        lam: float = 1.0,
        deep_mmd_loss_weight: float = 10.0,
        flatten_feature_extraction_layers: Dict[str, bool] = {},
        size_feature_extraction_layers: Dict[str, int] = {},
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
                from them what is the flattened feature size. Defaults to {}. If it is -1 then the layer is not
                flattened.
        """
        super().__init__(
            data_path=data_path,
            metrics=metrics,
            device=device,
            loss_meter_type=loss_meter_type,
            checkpointer=checkpointer,
            lam=lam,
        )
        self.deep_mmd_loss_weight = deep_mmd_loss_weight
        if self.deep_mmd_loss_weight == 0:
            log(
                ERROR,
                "DEEP MMD loss weight is set to 0. As MK-MMD loss will not be computed, ",
                "please use vanilla DittoClient instead.",
            )

        self.flatten_feature_extraction_layers = flatten_feature_extraction_layers
        self.size_feature_extraction_layers = size_feature_extraction_layers
        self.deep_mmd_losses = {}
        for layer in self.flatten_feature_extraction_layers.keys():
            self.deep_mmd_losses[layer] = DeepMmdLoss(
                device=self.device,
                input_size=self.size_feature_extraction_layers[layer],
                layer_name=layer,
            ).to(self.device)

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
        if self.deep_mmd_loss_weight != 0:
            # Compute the features of the init_global_model
            _ = self.init_global_model(input)
            init_global_features = self.init_global_feature_extractor.get_extracted_features()
            for key in init_global_features.keys():
                features[" ".join(["init_global", key])] = init_global_features[key]

        return {"global": global_preds, "local": local_preds}, features

    def _maybe_checkpoint(self, loss: float, metrics: Dict[str, Scalar], checkpoint_mode: CheckpointMode) -> None:
        # Hooks need to be removed before checkpointing the model
        self.local_feature_extractor.remove_hooks()
        super()._maybe_checkpoint(loss=loss, metrics=metrics, checkpoint_mode=checkpoint_mode)

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

        if self.deep_mmd_loss_weight != 0:
            # Compute DEEP-MMD loss
            total_deep_mmd_loss = torch.tensor(0.0, device=self.device)
            for layer in self.flatten_feature_extraction_layers.keys():
                layer_deep_mmd_loss = self.deep_mmd_losses[layer](
                    features[layer], features[" ".join(["init_global", layer])]
                )
                additional_losses["_".join(["mkmmd_loss", layer])] = layer_deep_mmd_loss
                total_deep_mmd_loss += layer_deep_mmd_loss
            total_loss += self.deep_mmd_loss_weight * total_deep_mmd_loss
            additional_losses["deep_mmd_loss_total"] = total_deep_mmd_loss
        additional_losses["total_loss"] = total_loss

        return total_loss, additional_losses
