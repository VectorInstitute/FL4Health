from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import torch
from flwr.common.typing import Config

from fl4health.checkpointing.checkpointer import TorchCheckpointer
from fl4health.clients.basic_client import BasicClient
from fl4health.losses.contrastive_loss import ContrastiveLoss
from fl4health.losses.mkmmd_loss import MkMmdLoss
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
        )
        """This module is used to init fenda client with various auxiliary loss functions.
        These losses will be activated only when their weights are not 0.0.
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
            mkmmd_loss_weights: Weights to be used for mkmmd losses. Fist value is for minimizing the distance
            and second value is for maximizing the distance.
        """
        self.perfcl_loss_weights = perfcl_loss_weights
        self.cos_sim_loss_weight = cos_sim_loss_weight
        self.contrastive_loss_weight = contrastive_loss_weight
        self.mkmmd_loss_weights = mkmmd_loss_weights
        self.cos_sim = torch.nn.CosineSimilarity(dim=-1).to(self.device)
        self.contrastive_loss = ContrastiveLoss(self.device, temperature=temperature).to(self.device)
        self.mkmmd_loss_min = MkMmdLoss(device=self.device, minimize_type_two_error=True).to(self.device)
        self.mkmmd_loss_max = MkMmdLoss(device=self.device, minimize_type_two_error=False).to(self.device)

        # Need to save previous local module, global module and aggregated global module at each communication round
        # to compute contrastive loss.
        self.old_local_module: Optional[torch.nn.Module] = None
        self.old_global_module: Optional[torch.nn.Module] = None
        self.aggregated_global_module: Optional[torch.nn.Module] = None

        self.local_buffer: list[torch.Tensor] = []
        self.global_buffer: list[torch.Tensor] = []
        self.aggregated_buffer: list[torch.Tensor] = []

    def get_parameter_exchanger(self, config: Config) -> ParameterExchanger:
        assert isinstance(self.model, FendaModel)
        return FixedLayerExchanger(self.model.layers_to_exchange())

    def predict(self, input: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Computes the prediction(s) and features of the model(s) given the input.

        Args:
            input (torch.Tensor): Inputs to be fed into the model.

        Returns:
            Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]: A tuple in which the first element
            contains predictions indexed by name and the second element contains intermediate activations
            index by name. Specificaly the features of the model, features of the global model and features of
            the old model are returned. All predictions included in dictionary will be used to compute metrics.
        """
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
        elif self.mkmmd_loss_weights is not None:
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
        if (
            self.mkmmd_loss_weights
            and self.old_global_module
            and self.old_local_module
            and self.aggregated_global_module
        ):
            self.update_buffers(
                local_module=self.old_local_module,
                global_module=self.old_global_module,
                aggregated_module=self.aggregated_global_module,
            )
            if self.mkmmd_loss_weights[0] != 0.0:
                self.mkmmd_loss_min.betas = self.mkmmd_loss_min.optimize_betas(
                    X=torch.cat(self.global_buffer, dim=0), Y=torch.cat(self.aggregated_buffer, dim=0), lambda_m=1e-5
                )
            if self.mkmmd_loss_weights[1] != 0.0:
                self.mkmmd_loss_max.betas = self.mkmmd_loss_max.optimize_betas(
                    X=torch.cat(self.local_buffer, dim=0), Y=torch.cat(self.aggregated_buffer, dim=0), lambda_m=1e-5
                )

            self.local_buffer.clear()
            self.global_buffer.clear()
            self.aggregated_buffer.clear()

        return super().update_before_train(current_server_round)

    def update_buffers(
        self, local_module: torch.nn.Module, global_module: torch.nn.Module, aggregated_module: torch.nn.Module
    ) -> None:
        """Update the feature buffer of the local, global and aggregated features."""
        assert isinstance(local_module, torch.nn.Module)
        assert isinstance(global_module, torch.nn.Module)
        assert isinstance(aggregated_module, torch.nn.Module)

        self.local_buffer.clear()
        self.global_buffer.clear()
        self.aggregated_buffer.clear()

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
                self.local_buffer.append(local_features.reshape(len(local_features), -1))
                self.global_buffer.append(global_features.reshape(len(global_features), -1))
                self.aggregated_buffer.append(aggregated_features.reshape(len(aggregated_features), -1))

    def get_cosine_similarity_loss(self, local_features: torch.Tensor, global_features: torch.Tensor) -> torch.Tensor:
        """
        Cosine similarity loss aims to minimize the similarity among current local features and current global
        features of fenda model.
        """
        assert len(local_features) == len(global_features)
        return torch.abs(self.cos_sim(local_features, global_features)).mean()

    def get_contrastive_loss(
        self, local_features: torch.Tensor, old_local_features: torch.Tensor, global_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute contrastive over current local features (z_p) with old local features (hat{z_p}) as positive pairs
        and current global features (z_s) as negative pairs.
        """
        return self.contrastive_loss(
            features=local_features,  # (z_p)
            positive_pairs=old_local_features.unsqueeze(0),  # (\hat{z_p})
            negative_pairs=global_features.unsqueeze(0),  # (z_s)
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
        Perfcl loss implemented based on https://www.sciencedirect.com/science/article/pii/S0031320323002078.

        This paper introduced two contrastive loss functions:
        1- First one aims to enhance the similarity between the current global features (z_s) and aggregated global
        features (z_g) as positive pairs while reducing the similarity between the current global features (z_s)
        and old global features (hat{z_s}) as negative pairs.
        2- Second one aims to enhance the similarity between the current local features (z_p) and old local features
        (hat{z_p}) as positive pairs while reducing the similarity between the current local features (z_p) and
        aggregated lobal features (z_g) as negative pairs.
        """

        global_contrastive_loss = self.contrastive_loss(
            features=global_features,  # (z_s)
            positive_pairs=aggregated_global_features.unsqueeze(0),  # (z_g)
            negative_pairs=old_global_features.unsqueeze(0),  # (\hat{z_s})
        )
        local_contrastive_loss = self.contrastive_loss(
            features=local_features,  # (z_p)
            positive_pairs=old_local_features.unsqueeze(0),  # (\hat{z_p})
            negative_pairs=aggregated_global_features.unsqueeze(0),  # (z_g)
        )

        return global_contrastive_loss, local_contrastive_loss

    def get_mkmmd_loss(
        self, mkmmd_loss: MkMmdLoss, distribution_a: torch.Tensor, distribution_b: torch.Tensor
    ) -> torch.Tensor:
        """
        MK-MMD loss aims to compute the distance among given two distributions with optimized betas.
        """
        assert distribution_a.shape == distribution_b.shape
        return mkmmd_loss(distribution_a, distribution_b)

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
            global_contrastive_loss, personal_contrastive_loss = self.get_perfcl_loss(
                local_features=features["local_features"],
                old_local_features=features["old_local_features"],
                global_features=features["global_features"],
                old_global_features=features["old_global_features"],
                aggregated_global_features=features["aggregated_global_features"],
            )
            total_loss += (
                self.perfcl_loss_weights[0] * global_contrastive_loss
                + self.perfcl_loss_weights[1] * personal_contrastive_loss
            )
            additional_losses["global_contrastive_loss"] = global_contrastive_loss
            additional_losses["personal_contrastive_loss"] = personal_contrastive_loss

        if self.mkmmd_loss_weights:
            if self.mkmmd_loss_weights[0] != 0.0:
                mkmmd_loss_min = self.get_mkmmd_loss(
                    mkmmd_loss=self.mkmmd_loss_min,
                    distribution_a=features["global_features"],
                    distribution_b=features["aggregated_global_features"],
                )
                total_loss += self.mkmmd_loss_weights[0] * mkmmd_loss_min.sum()
                additional_losses["mkmmd_loss_min"] = mkmmd_loss_min

            if self.mkmmd_loss_weights[1] != 0.0:
                mkmmd_loss_max = self.get_mkmmd_loss(
                    mkmmd_loss=self.mkmmd_loss_max,
                    distribution_a=features["local_features"],
                    distribution_b=features["aggregated_global_features"],
                )
                total_loss -= self.mkmmd_loss_weights[1] * mkmmd_loss_max.sum()
                additional_losses["mkmmd_loss_max"] = mkmmd_loss_max

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
