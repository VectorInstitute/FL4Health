from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import torch
from flwr.common.typing import Config

from fl4health.checkpointing.client_module import ClientCheckpointModule
from fl4health.clients.basic_client import BasicClient, TorchInputType
from fl4health.model_bases.fenda_base import FendaModel
from fl4health.parameter_exchange.layer_exchanger import FixedLayerExchanger
from fl4health.parameter_exchange.parameter_exchanger_base import ParameterExchanger
from fl4health.utils.losses import EvaluationLosses, LossMeterType
from fl4health.utils.metrics import Metric


class FendaClient(BasicClient):
    def __init__(
        self,
        data_path: Path,
        metrics: Sequence[Metric],
        device: torch.device,
        loss_meter_type: LossMeterType = LossMeterType.AVERAGE,
        checkpointer: Optional[ClientCheckpointModule] = None,
        temperature: Optional[float] = 0.5,
        perfcl_loss_weights: Optional[Tuple[float, float]] = None,
        cos_sim_loss_weight: Optional[float] = None,
        contrastive_loss_weight: Optional[float] = None,
    ) -> None:
        super().__init__(
            data_path=data_path,
            metrics=metrics,
            device=device,
            loss_meter_type=loss_meter_type,
            checkpointer=checkpointer,
        )
        """This module is used to init FENDA client with various auxiliary loss functions.
        These losses will be activated only when their weights are not 0.0.
        Args:
            data_path: Path to the data directory.
            metrics: List of metrics to be used for evaluation.
            device: Device to be used for training.
            loss_meter_type: Type of loss meter to be used.
            checkpointer: Checkpointer to be used for checkpointing.
            temperature: Temperature to be used for contrastive loss.
            perfcl_loss_weights: Weights to be used for PerFCL loss.
            Each value associate with one of two contrastive losses in PerFCL loss.
            cos_sim_loss_weight: Weight to be used for cosine similarity loss.
            contrastive_loss_weight: Weight to be used for contrastive loss.
        """
        self.perfcl_loss_weights = perfcl_loss_weights
        self.cos_sim_loss_weight = cos_sim_loss_weight
        self.contrastive_loss_weight = contrastive_loss_weight
        self.cos_sim = torch.nn.CosineSimilarity(dim=-1).to(self.device)
        self.ce_criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.temperature = temperature

        # Need to save previous local module, global module and aggregated global module at each communication round
        # to compute contrastive loss.
        self.old_local_module: Optional[torch.nn.Module] = None
        self.old_global_module: Optional[torch.nn.Module] = None
        self.aggregated_global_module: Optional[torch.nn.Module] = None

    def get_parameter_exchanger(self, config: Config) -> ParameterExchanger:
        assert isinstance(self.model, FendaModel)
        return FixedLayerExchanger(self.model.layers_to_exchange())

    def predict(self, input: TorchInputType) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
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
        preds, features = self.model(input)
        if self.contrastive_loss_weight or self.perfcl_loss_weights:
            if self.old_local_module is not None:
                features["old_local_features"] = self.old_local_module.forward(input).reshape(
                    len(features["local_features"]), -1
                )
                if self.perfcl_loss_weights:
                    if self.old_global_module is not None:
                        features["old_global_features"] = self.old_global_module.forward(input).reshape(
                            len(features["global_features"]), -1
                        )
                    if self.aggregated_global_module is not None:
                        features["aggregated_global_features"] = self.aggregated_global_module.forward(input).reshape(
                            len(features["global_features"]), -1
                        )
        return preds, features

    def update_after_train(self, local_steps: int, loss_dict: Dict[str, float]) -> None:
        # Save the parameters of the old model
        assert isinstance(self.model, FendaModel)
        if self.contrastive_loss_weight or self.perfcl_loss_weights:
            self.old_local_module = self.clone_and_freeze_model(self.model.local_module)
            self.old_global_module = self.clone_and_freeze_model(self.model.global_module)

        return super().update_after_train(local_steps, loss_dict)

    def update_before_train(self, current_server_round: int) -> None:
        # Save the parameters of the aggregated global model
        assert isinstance(self.model, FendaModel)
        if self.perfcl_loss_weights:
            self.aggregated_global_module = self.clone_and_freeze_model(self.model.global_module)
        return super().update_before_train(current_server_round)

    def get_cosine_similarity_loss(self, local_features: torch.Tensor, global_features: torch.Tensor) -> torch.Tensor:
        """
        Cosine similarity loss aims to minimize the similarity among current local features and current global
        features of FENDA model.
        """
        assert len(local_features) == len(global_features)
        return torch.abs(self.cos_sim(local_features, global_features)).mean()

    def compute_contrastive_loss(
        self, features: torch.Tensor, positive_pairs: torch.Tensor, negative_pairs: torch.Tensor
    ) -> torch.Tensor:
        """
        Contrastive loss aims to enhance the similarity between the features and their positive pairs
        while reducing the similarity between the features and their negative pairs.
        """
        assert self.temperature is not None
        assert len(features) == len(positive_pairs)
        posi = self.cos_sim(features, positive_pairs)
        logits = posi.reshape(-1, 1)
        assert len(features) == len(negative_pairs)
        nega = self.cos_sim(features, negative_pairs)
        logits = torch.cat((logits, nega.reshape(-1, 1)), dim=1)
        logits /= self.temperature
        labels = torch.zeros(features.size(0)).to(self.device).long()

        return self.ce_criterion(logits, labels)

    def get_contrastive_loss(
        self, local_features: torch.Tensor, old_local_features: torch.Tensor, global_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute contrastive over current local features (z_p) with old local features (hat{z_p}) as positive pairs
        and current global features (z_s) as negative pairs.
        """
        return self.compute_contrastive_loss(
            features=local_features,  # (z_p)
            positive_pairs=old_local_features,  # (\hat{z_p})
            negative_pairs=global_features,  # (z_s)
        )

    def get_perfcl_loss(
        self,
        local_features: torch.Tensor,
        old_local_features: torch.Tensor,
        global_features: torch.Tensor,
        old_global_features: torch.Tensor,
        aggregated_global_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        PerFCL loss implemented based on https://www.sciencedirect.com/science/article/pii/S0031320323002078.

        This paper introduced two contrastive loss functions:
        1- First one aims to enhance the similarity between the current global features (z_s) and aggregated global
        features (z_g) as positive pairs while reducing the similarity between the current global features (z_s)
        and old global features (hat{z_s}) as negative pairs.
        2- Second one aims to enhance the similarity between the current local features (z_p) and old local features
        (hat{z_p}) as positive pairs while reducing the similarity between the current local features (z_p) and
        aggregated global features (z_g) as negative pairs.
        """

        contrastive_loss_minimize = self.compute_contrastive_loss(
            features=global_features,  # (z_s)
            positive_pairs=aggregated_global_features,  # (z_g)
            negative_pairs=old_global_features,  # (\hat{z_s})
        )
        contrastive_loss_maximize = self.compute_contrastive_loss(
            features=local_features,  # (z_p)
            positive_pairs=old_local_features,  # (\hat{z_p})
            negative_pairs=aggregated_global_features,  # (z_g)
        )

        return contrastive_loss_minimize, contrastive_loss_maximize

    def compute_loss_and_additional_losses(
        self,
        preds: Dict[str, torch.Tensor],
        features: Dict[str, torch.Tensor],
        target: torch.Tensor,
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

        # Optimal cos_sim_loss_weight for FedIsic dataset is 100.0
        if self.cos_sim_loss_weight:
            cos_sim_loss = self.get_cosine_similarity_loss(
                local_features=features["local_features"],
                global_features=features["global_features"],
            )
            total_loss += self.cos_sim_loss_weight * cos_sim_loss
            additional_losses["cos_sim_loss"] = cos_sim_loss

        # Optimal contrastive_loss_weight for FedIsic dataset is 10.0
        if self.contrastive_loss_weight and "old_local_features" in features:
            contrastive_loss = self.get_contrastive_loss(
                local_features=features["local_features"],
                old_local_features=features["old_local_features"],
                global_features=features["global_features"],
            )
            total_loss += self.contrastive_loss_weight * contrastive_loss
            additional_losses["contrastive_loss"] = contrastive_loss

        # Optimal perfcl_loss_weights for FedIsic dataset is [10.0, 10.0]
        if self.perfcl_loss_weights and "old_local_features" in features and "old_global_features" in features:
            contrastive_loss_minimize, contrastive_loss_maximize = self.get_perfcl_loss(
                local_features=features["local_features"],
                old_local_features=features["old_local_features"],
                global_features=features["global_features"],
                old_global_features=features["old_global_features"],
                aggregated_global_features=features["aggregated_global_features"],
            )
            total_loss += (
                self.perfcl_loss_weights[0] * contrastive_loss_minimize
                + self.perfcl_loss_weights[1] * contrastive_loss_maximize
            )
            additional_losses["contrastive_loss_minimize"] = contrastive_loss_minimize
            additional_losses["contrastive_loss_maximize"] = contrastive_loss_maximize

        additional_losses["total_loss"] = total_loss

        return total_loss, additional_losses

    def compute_evaluation_loss(
        self,
        preds: Dict[str, torch.Tensor],
        features: Dict[str, torch.Tensor],
        target: torch.Tensor,
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
