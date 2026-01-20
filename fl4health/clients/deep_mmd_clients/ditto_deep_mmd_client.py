from collections.abc import Sequence
from logging import ERROR
from pathlib import Path

import torch
from flwr.common.logger import log
from flwr.common.typing import Config, Scalar
from torch import nn

from fl4health.checkpointing.client_module import CheckpointMode, ClientCheckpointAndStateModule
from fl4health.clients.ditto_client import DittoClient
from fl4health.losses.deep_mmd_loss import DeepMmdLoss
from fl4health.metrics.base_metrics import Metric
from fl4health.model_bases.feature_extractor_buffer import FeatureExtractorBuffer
from fl4health.reporting.base_reporter import BaseReporter
from fl4health.utils.client import clone_and_freeze_model
from fl4health.utils.losses import EvaluationLosses, LossMeterType, TrainingLosses
from fl4health.utils.random import restore_random_state, save_random_state
from fl4health.utils.typing import TorchFeatureType, TorchInputType, TorchPredType, TorchTargetType


class DittoDeepMmdClient(DittoClient):
    def __init__(
        self,
        data_path: Path,
        metrics: Sequence[Metric],
        device: torch.device,
        loss_meter_type: LossMeterType = LossMeterType.AVERAGE,
        checkpoint_and_state_module: ClientCheckpointAndStateModule | None = None,
        reporters: Sequence[BaseReporter] | None = None,
        progress_bar: bool = False,
        client_name: str | None = None,
        deep_mmd_loss_weight: float = 10.0,
        feature_extraction_layers_with_size: dict[str, int] | None = None,
        mmd_kernel_train_interval: int = 20,
        num_accumulating_batches: int | None = None,
    ) -> None:
        """
        This client implements the Deep MMD loss function in the Ditto framework. The Deep MMD loss is a measure of
        the distance between the distributions of the features of the local model and initial global model of each
        round. The Deep MMD loss is added to the local loss to penalize the local model for drifting away from the
        global model.

        Args:
            data_path (Path): path to the data to be used to load the data for client-side training.
            metrics (Sequence[Metric]): Metrics to be computed based on the labels and predictions of the client model.
            device (torch.device): Device indicator for where to send the model, batches, labels etc. Often "cpu" or
                "cuda".
            loss_meter_type (LossMeterType, optional): Type of meter used to track and compute the losses over
                each batch. Defaults to ``LossMeterType.AVERAGE``.
            checkpoint_and_state_module (ClientCheckpointAndStateModule | None, optional): A module meant to handle
                both checkpointing and state saving. The module, and its underlying model and state checkpointing
                components will determine when and how to do checkpointing during client-side training.
                No checkpointing (state or model) is done if not provided. Defaults to None.
            reporters (Sequence[BaseReporter] | None, optional): A sequence of FL4Health reporters which the client
                should send data to. Defaults to None.
            progress_bar (bool, optional): Whether or not to display a progress bar during client training and
                validation. Uses ``tqdm``. Defaults to False
            client_name (str | None, optional): An optional client name that uniquely identifies a client.
                If not passed, a hash is randomly generated. Client state will use this as part of its state file
                name. Defaults to None.
            deep_mmd_loss_weight (float, optional): weight applied to the Deep MMD loss. Defaults to 10.0.
            feature_extraction_layers_with_size (dict[str, int] | None, optional): Dictionary of layers to extract
                features from them and their respective feature size. Defaults to None.
            mmd_kernel_train_interval (int, optional): interval at which to train and update the Deep MMD kernel. If
                set to above 0, the kernel will be train based on whole distribution of latent features of data with
                the given train interval. If set to 0, the kernel will not be trained. If set to -1, the kernel will
                be trained after each individual batch based on only that individual batch. Defaults to 20.
            num_accumulating_batches (int, optional): Number of batches to accumulate features to approximate the whole
                distribution of the latent features for updating Deep MMD kernel. This parameter is only used
                if ``mmd_kernel_train_interval`` is set to larger than 0. Defaults to None.
        """
        super().__init__(
            data_path=data_path,
            metrics=metrics,
            device=device,
            loss_meter_type=loss_meter_type,
            checkpoint_and_state_module=checkpoint_and_state_module,
            reporters=reporters,
            progress_bar=progress_bar,
            client_name=client_name,
        )
        self.deep_mmd_loss_weight = deep_mmd_loss_weight
        if self.deep_mmd_loss_weight == 0:
            log(
                ERROR,
                "Deep MMD loss weight is set to 0. As Deep MMD loss will not be computed, ",
                "please use vanilla DittoClient instead.",
            )

        if feature_extraction_layers_with_size is None:
            feature_extraction_layers_with_size = {}
        self.flatten_feature_extraction_layers = dict.fromkeys(feature_extraction_layers_with_size.keys(), True)
        self.deep_mmd_losses: dict[str, DeepMmdLoss] = {}
        # Save the random state to be restored after initializing the Deep MMD loss layers.
        random_state, numpy_state, torch_state = save_random_state()
        for layer, feature_size in feature_extraction_layers_with_size.items():
            self.deep_mmd_losses[layer] = DeepMmdLoss(
                device=self.device,
                input_size=feature_size,
            ).to(self.device)
        # Restore the random state after initializing the Deep MMD loss layers. This is to ensure that the random state
        # would not change after initializing the Deep MMD loss.
        restore_random_state(random_state, numpy_state, torch_state)
        self.initial_global_model: nn.Module
        self.local_feature_extractor: FeatureExtractorBuffer
        self.initial_global_feature_extractor: FeatureExtractorBuffer
        self.num_accumulating_batches = num_accumulating_batches
        self.mmd_kernel_train_interval = mmd_kernel_train_interval

    def setup_client(self, config: Config) -> None:
        super().setup_client(config)
        self.local_feature_extractor = FeatureExtractorBuffer(
            model=self.model,
            flatten_feature_extraction_layers=self.flatten_feature_extraction_layers,
        )
        # Register hooks to extract features from the local model if not already registered
        self.local_feature_extractor._maybe_register_hooks()

    def update_before_train(self, current_server_round: int) -> None:
        super().update_before_train(current_server_round)
        assert isinstance(self.global_model, nn.Module)
        # Clone and freeze the initial weights GLOBAL MODEL. These are used to form the Ditto local
        # update penalty term.
        self.initial_global_model = clone_and_freeze_model(self.global_model)
        self.initial_global_feature_extractor = FeatureExtractorBuffer(
            model=self.initial_global_model,
            flatten_feature_extraction_layers=self.flatten_feature_extraction_layers,
        )
        # Register hooks to extract features from the initial global model if not already registered
        self.initial_global_feature_extractor._maybe_register_hooks()
        # Enable training of Deep MMD loss layers if the mmd_kernel_train_interval is set to -1
        # meaning that the kernel parameters will be trained after each individual batch based on only that
        # individual batch
        if self.mmd_kernel_train_interval == -1:
            for layer in self.flatten_feature_extraction_layers:
                self.deep_mmd_losses[layer].training = True

    def _should_optimize_betas(self, step: int) -> bool:
        step_at_interval = (step - 1) % self.mmd_kernel_train_interval == 0
        valid_components_present = self.initial_global_model is not None
        # If the Deep MMD loss doesn't matter, we don't bother optimizing betas
        weighted_deep_mmd_loss = self.deep_mmd_loss_weight != 0
        return step_at_interval and valid_components_present and weighted_deep_mmd_loss

    def update_after_step(self, step: int, current_round: int | None = None) -> None:
        if self.mmd_kernel_train_interval > 0 and self._should_optimize_betas(step):
            # Get the feature distribution of the local and initial global features with evaluation
            # mode
            local_distributions, initial_global_distributions = self.update_buffers(
                self.model, self.initial_global_model
            )
            # As we set the training mode of the Deep MMD loss layers to True, we train the
            # kernel of the Deep MMD loss based on gathered features in the buffer and compute the
            # Deep MMD loss
            for layer, layer_deep_mmd_loss in self.deep_mmd_losses.items():
                layer_deep_mmd_loss.training = True
                layer_deep_mmd_loss(local_distributions[layer], initial_global_distributions[layer])
                layer_deep_mmd_loss.training = False
        super().update_after_step(step)

    def update_buffers(
        self, local_model: torch.nn.Module, initial_global_model: torch.nn.Module
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """
        Update the feature buffer of the local and global features.

        Args:
            local_model (torch.nn.Module): Local model to extract features from.
            initial_global_model (torch.nn.Module): Initial global model to extract features from.

        Returns:
            (tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]): A tuple containing the extracted
                features using the local and initial global models.
        """
        self.local_feature_extractor.clear_buffers()
        self.initial_global_feature_extractor.clear_buffers()

        self.local_feature_extractor.enable_accumulating_features()
        self.initial_global_feature_extractor.enable_accumulating_features()

        # Save the initial state of the local model to restore it after the buffer is populated,
        # however as initial global model is already cloned and frozen, we don't need to save its state.
        initial_state_local_model = local_model.training

        # Set local model to evaluation mode, as we don't want to create a computational graph
        # for the local model when populating the local buffer with features to train Deep MMD
        # kernel
        local_model.eval()

        # Make sure the local model is in evaluation mode before populating the local buffer
        assert not local_model.training

        # Make sure the initial global model is in evaluation mode before populating the global buffer
        # as it is already cloned and frozen from the global model
        assert not initial_global_model.training

        with torch.no_grad():
            for i, (input, _) in enumerate(self.train_loader):
                input = input.to(self.device)
                # Pass the input through the local model to populate the local_feature_extractor buffer
                local_model(input)
                # Pass the input through the initial global model to populate the initial_global_feature_extractor
                # buffer
                initial_global_model(input)
                # Break if the number of accumulating batches is reached to avoid memory issues
                if i == self.num_accumulating_batches:
                    break
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
    ) -> tuple[TorchPredType, TorchFeatureType]:
        """
        Computes the predictions for both the **GLOBAL** and **LOCAL** models and pack them into the prediction
        dictionary.

        Args:
            input (TorchInputType): Inputs to be fed into the model. If input is of type ``dict[str, torch.Tensor]``,
                it is assumed that the keys of input match the names of the keyword arguments of
                ``self.model.forward()``.

        Returns:
            (tuple[TorchPredType, TorchFeatureType]): A tuple in which the first element contains a dictionary of
                predictions indexed by name and the second element contains intermediate activations indexed by name.
                By passing features, we can compute all the losses. All predictions included in dictionary will by
                default be used to compute metrics separately.
        """
        # We use features from initial_global_model to compute the Deep MMD loss not the global_model
        global_preds = self.global_model(input)
        local_preds = self.model(input)
        features = self.local_feature_extractor.get_extracted_features()
        if self.deep_mmd_loss_weight != 0:
            # Compute the features of the initial_global_model
            self.initial_global_model(input)
            initial_global_features = self.initial_global_feature_extractor.get_extracted_features()
            for key, initial_global_feature in initial_global_features.items():
                features[" ".join(["init_global", key])] = initial_global_feature

        return {"global": global_preds, "local": local_preds}, features

    def _maybe_checkpoint(self, loss: float, metrics: dict[str, Scalar], checkpoint_mode: CheckpointMode) -> None:
        # Hooks need to be removed before checkpointing the model
        self.local_feature_extractor.remove_hooks()
        super()._maybe_checkpoint(loss=loss, metrics=metrics, checkpoint_mode=checkpoint_mode)
        # As hooks have to be removed to checkpoint the model, so we check if they need to be re-registered
        # each time.
        self.local_feature_extractor._maybe_register_hooks()

    def validate(self, include_losses_in_metrics: bool = False) -> tuple[float, dict[str, Scalar]]:
        """
        Validate the current model on the entire validation dataset.

        Returns:
            (tuple[float, dict[str, Scalar]]): The validation loss and a dictionary of metrics from validation.
        """
        for layer in self.flatten_feature_extraction_layers:
            self.deep_mmd_losses[layer].training = False
        return super().validate(include_losses_in_metrics)

    def compute_training_loss(
        self,
        preds: TorchPredType,
        features: TorchFeatureType,
        target: TorchTargetType,
    ) -> TrainingLosses:
        """
        Computes training losses given predictions of the global and local models and ground truth data.
        For the local model we add to the vanilla loss function by including Ditto penalty loss which is the
        \\(\\ell^2\\) inner product between the initial global model weights and weights of the local model. This is
        stored in backward The loss to optimize the global model is stored in the additional losses dictionary
        under “global_loss”.

        Args:
            preds (TorchPredType): Prediction(s) of the model(s) indexed by name. All predictions included in
                dictionary will be used to compute metrics.
            features (TorchFeatureType): Feature(s) of the model(s) indexed by name.
            target (TorchTargetType): Ground truth data to evaluate predictions against.

        Returns:
            (TrainingLosses): An instance of ``TrainingLosses`` containing backward loss and additional losses indexed
                by name. Additional losses includes each loss component and the global model loss tensor.
        """
        for layer_loss_module in self.deep_mmd_losses.values():
            if self.mmd_kernel_train_interval == -1:
                assert layer_loss_module.training
            else:
                assert not layer_loss_module.training
        # Check that both models are in training mode
        assert self.global_model.training and self.model.training

        # local loss is stored in loss, global model loss is stored in additional losses.
        loss, additional_losses = self.compute_loss_and_additional_losses(preds, features, target)

        # Setting the adaptation loss to that of the local model, as its performance should dictate whether more or
        # less weight is used to constrain it to the global model (as in FedProx)
        additional_losses["loss_for_adaptation"] = additional_losses["local_loss"].clone()

        # This is the Ditto penalty loss of the local model compared with the original Global model weights, scaled
        # by drift_penalty_weight (or lambda in the original paper)
        penalty_loss = self.compute_penalty_loss()
        additional_losses["penalty_loss"] = penalty_loss.clone()
        total_loss = loss + penalty_loss

        # Add Deep MMD loss based on computed features during training
        if self.deep_mmd_loss_weight != 0:
            total_loss += additional_losses["deep_mmd_loss_total"]

        additional_losses["total_loss"] = total_loss.clone()

        return TrainingLosses(backward=total_loss, additional_losses=additional_losses)

    def compute_evaluation_loss(
        self,
        preds: TorchPredType,
        features: TorchFeatureType,
        target: TorchTargetType,
    ) -> EvaluationLosses:
        """
        Computes evaluation loss given predictions (and potentially features) of the model and ground truth data.

        Args:
            preds (TorchPredType): Prediction(s) of the model(s) indexed by name. Anything stored in preds will be
                used to compute metrics.
            features (TorchFeatureType): Feature(s) of the model(s) indexed by name.
            target (TorchTargetType): Ground truth data to evaluate predictions against.

        Returns:
            (EvaluationLosses): An instance of ``EvaluationLosses`` containing checkpoint loss and additional losses
                indexed by name.
        """
        for layer_loss_module in self.deep_mmd_losses.values():
            assert not layer_loss_module.training
        return super().compute_evaluation_loss(preds, features, target)

    def compute_loss_and_additional_losses(
        self, preds: TorchPredType, features: TorchFeatureType, target: TorchTargetType
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Computes the loss and any additional losses given predictions of the model and ground truth data.

        Args:
            preds (TorchPredType): Prediction(s) of the model(s) indexed by name.
            features (TorchFeatureType): Feature(s) of the model(s) indexed by name.
            target (TorchTargetType): Ground truth data to evaluate predictions against.

        Returns:
            (tuple[torch.Tensor, dict[str, torch.Tensor]]): A tuple with:

                - The tensor for the loss
                - A dictionary of additional losses with their names and values, or None if there are no additional
                  losses.
        """
        loss, additional_losses = super().compute_loss_and_additional_losses(preds, features, target)

        if self.deep_mmd_loss_weight != 0:
            # Compute Deep MMD loss based on computed features during training
            total_deep_mmd_loss = torch.tensor(0.0, device=self.device)
            for layer, layer_deep_mmd_loss in self.deep_mmd_losses.items():
                deep_mmd_loss = layer_deep_mmd_loss(features[layer], features[" ".join(["init_global", layer])])
                additional_losses["_".join(["deep_mmd_loss", layer])] = deep_mmd_loss.clone()
                total_deep_mmd_loss += deep_mmd_loss
            additional_losses["deep_mmd_loss_total"] = self.deep_mmd_loss_weight * total_deep_mmd_loss

        return loss, additional_losses
