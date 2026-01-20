import warnings
from collections.abc import Sequence
from logging import WARN
from pathlib import Path
from typing import Any

import torch
from flwr.common.logger import log
from torch import nn
from torch.optim import Optimizer

from fl4health.checkpointing.client_module import ClientCheckpointAndStateModule
from fl4health.clients.basic_client import BasicClient
from fl4health.metrics.base_metrics import Metric
from fl4health.reporting.base_reporter import BaseReporter
from fl4health.utils.losses import EvaluationLosses, LossMeterType, TrainingLosses
from fl4health.utils.typing import (
    TorchFeatureType,
    TorchInputType,
    TorchPredType,
    TorchTargetType,
)


EXPECTED_OUTPUT_TUPLE_SIZE = 2


class FlexibleClient(BasicClient):
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
    ) -> None:
        """
        Flexible FL Client with functionality to train, evaluate, log, report and checkpoint.

        ``FlexibleClient`` is similar to ``BasicClient`` but provides added flexibility through the
        ability to inject models and optimizers in the methods responsible for making predictions
        and performing both train and validation steps.

        This added flexibility allows for ``FlexibleClient`` to be automatically adapted with our
        personalized methods: ``~fl4health.mixins.personalized``.

        As with ``BasicClient``, users are responsible for implementing methods:

            - ``get_model``
            - ``get_optimizer``
            - ``get_data_loaders``,
            - ``get_criterion``

        However, unlike ``BasicClient``, users looking to specialize logic for making predictions,
        and performing train and validation steps, should instead override:

            - ``predict_with_model``
            - ``_train_step_with_model_and_optimizer`` (and its delegated helpers)
            - ``_val_step_with_model``

        Other methods can be overridden to achieve custom functionality.

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
        """
        super().__init__(
            data_path,
            metrics,
            device,
            loss_meter_type,
            checkpoint_and_state_module,
            reporters,
            progress_bar,
            client_name,
        )

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Perform some validations on subclasses of FlexibleClient."""
        super().__init_subclass__(**kwargs)

        # check that specific methods are not overridden, otherwise throw warning
        methods_should_not_be_overridden = [
            (
                "predict",
                (
                    f"`{cls.__name__}` overrides `predict()`, but this method should no longer be overridden. "
                    "Please use `predict_with_model()` instead."
                ),
            ),
            (
                "val_step",
                (
                    f"`{cls.__name__}` overrides `val_step()`, but this method should no longer be overridden. "
                    "Please use `_val_step_with_model()` instead."
                ),
            ),
            (
                "train_step",
                (
                    f"`{cls.__name__}` overrides `train_step()`, but this method should no longer be overridden. "
                    "Please use `_train_step_with_model_and_optimizer()` and its helper methods instead "
                    "for proper customization."
                ),
            ),
        ]

        for method_name, msg in methods_should_not_be_overridden:
            if method_name in cls.__dict__:  # method was overridden by subclass
                log(WARN, msg)
                warnings.warn(msg, RuntimeWarning, stacklevel=2)

    def _compute_preds_and_losses(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        input: TorchInputType,
        target: TorchTargetType,
    ) -> tuple[TrainingLosses, TorchPredType]:
        """
        Helper method within the train step for computing preds and losses.

        **NOTE**: Subclasses should implement this helper method if there is a need
        to specialize this part of the overall train step.

        Args:
            model (nn.Module): the model used to make predictions
            optimizer (Optimizer): the associated optimizer
            input (TorchInputType): The input to be fed into the model.
            target (TorchTargetType): The target corresponding to the input.

        Returns:
            (tuple[TrainingLosses, TorchPredType]): The losses object from the train step along with a dictionary of
                any predictions produced by the model prior to the application of the backwards phase
        """
        # Clear gradients from optimizer if they exist
        optimizer.zero_grad()

        # Call user defined methods to get predictions and compute loss
        preds, features = self.predict_with_model(model, input)
        target = self.transform_target(target)
        losses = self.compute_training_loss(preds, features, target)

        return losses, preds

    def _apply_backwards_on_losses_and_take_step(
        self, model: nn.Module, optimizer: Optimizer, losses: TrainingLosses
    ) -> TrainingLosses:
        """
        Helper method within the train step for applying backwards on losses and taking step with optimizer.

        **NOTE**: Subclasses should implement this helper method if there is a need
        to specialize this part of the overall train step.

        Args:
            model (nn.Module): the model used for making predictions. Passed here in case subclasses need it.
            optimizer (Optimizer): the optimizer with which we take the step
            losses (TrainingLosses): the losses to apply backwards on

        Returns:
            (TrainingLosses): The losses object post backwards application
        """
        # Compute backward pass and update parameters with optimizer
        losses.backward["backward"].backward()
        self._transform_gradients_with_model(model, losses)
        optimizer.step()

        return losses

    def _train_step_with_model_and_optimizer(
        self,
        model: torch.nn.Module,
        optimizer: Optimizer,
        input: TorchInputType,
        target: TorchTargetType,
    ) -> tuple[TrainingLosses, TorchPredType]:
        """
        Helper train step method that allows for injection of model and optimizer.

        **NOTE**: Subclasses should implement this method if there is a need to specialize
        the train_step logic.

        Args:
            model (nn.Module): the model used for making predictions. Passed here in case subclasses need it.
            optimizer (Optimizer): the optimizer with which we take the step
            input (TorchInputType): The input to be fed into the model.
            target (TorchTargetType): The target corresponding to the input.

        Returns:
            (tuple[TrainingLosses, TorchPredType]): The losses object from the train step along with
                a dictionary of any predictions produced by the model.
        """
        losses, preds = self._compute_preds_and_losses(model, optimizer, input, target)
        losses = self._apply_backwards_on_losses_and_take_step(model, optimizer, losses)

        return losses, preds

    def train_step(self, input: TorchInputType, target: TorchTargetType) -> tuple[TrainingLosses, TorchPredType]:
        """
        Given a single batch of input and target data, generate predictions, compute loss, update parameters and
        optionally update metrics if they exist. (i.e. backprop on a single batch of data).
        Assumes ``self.model`` is in train mode already.

        Args:
            input (TorchInputType): The input to be fed into the model.
            target (TorchTargetType): The target corresponding to the input.

        Returns:
            (tuple[TrainingLosses, TorchPredType]): The losses object from the train step along with
                a dictionary of any predictions produced by the model.
        """
        return self._train_step_with_model_and_optimizer(self.model, self.optimizers["global"], input, target)

    def _val_step_with_model(
        self, model: nn.Module, input: TorchInputType, target: TorchTargetType
    ) -> tuple[EvaluationLosses, TorchPredType]:
        """
        Helper method for val_step that allows for injection of model.

        **NOTE**: Subclasses should implement this method if there is a need to
        specialize the val_step logic.

        Args:
            model (nn.Module): the model used for making predictions. Passed here in case subclasses need it.
            input (TorchInputType): The input to be fed into the model.
            target (TorchTargetType): The target corresponding to the input.

        Returns:
            (tuple[EvaluationLosses, TorchPredType]): The losses object from the val step along with a dictionary of
                the predictions produced by the model.
        """
        # Get preds and compute loss
        with torch.no_grad():
            preds, features = self.predict_with_model(model, input)
            target = self.transform_target(target)
            losses = self.compute_evaluation_loss(preds, features, target)

        return losses, preds

    def val_step(self, input: TorchInputType, target: TorchTargetType) -> tuple[EvaluationLosses, TorchPredType]:
        """
        Given input and target, compute loss, update loss and metrics. Assumes ``self.model`` is in eval mode already.

        Args:
            input (TorchInputType): The input to be fed into the model.
            target (TorchTargetType): The target corresponding to the input.

        Returns:
            (tuple[EvaluationLosses, TorchPredType]): The losses object from the val step along with a dictionary of
                the predictions produced by the model.
        """
        return self._val_step_with_model(self.model, input, target)

    def predict_with_model(
        self, model: torch.nn.Module, input: TorchInputType
    ) -> tuple[TorchPredType, TorchFeatureType]:
        """
        Helper predict method that allows for injection of model.

        **NOTE**: Subclasses should implement this method if there is need to specialize
        the predict logic of the client.

        Args:
            model (torch.nn.Module): the model with which to make predictions
            input (TorchInputType): Inputs to be fed into the model. If input is of type ``dict[str, torch.Tensor]``,
                it is assumed that the keys of input match the names of the keyword arguments of
                ``self.model.forward().``

        Returns:
            (tuple[TorchPredType, TorchFeatureType]): A tuple in which the first element contains a dictionary of
                predictions indexed by name and the second element contains intermediate activations indexed by name.
                By passing features, we can compute losses such as the contrastive loss in MOON. All predictions
                included in dictionary will by default be used to compute metrics separately.

        Raises:
            TypeError: Occurs when something other than a tensor or dict of tensors is passed in to the model's
                forward method.
            ValueError: Occurs when something other than a tensor or dict of tensors is returned by the model
                forward.
        """
        if isinstance(input, torch.Tensor):
            output = model(input)
        elif isinstance(input, dict):
            # If input is a dictionary, then we unpack it before computing the forward pass.
            # Note that this assumes the keys of the input match (exactly) the keyword args
            # of self.model.forward().
            output = model(**input)
        else:
            raise TypeError("'input' must be of type torch.Tensor or dict[str, torch.Tensor].")

        if isinstance(output, dict):
            return output, {}
        if isinstance(output, torch.Tensor):
            return {"prediction": output}, {}
        if isinstance(output, tuple):
            if len(output) != EXPECTED_OUTPUT_TUPLE_SIZE:
                raise ValueError(f"Output tuple should have length 2 but has length {len(output)}")
            preds, features = output
            return preds, features
        raise ValueError("Model forward did not return a tensor, dictionary of tensors, or tuple of tensors")

    def _transform_gradients_with_model(self, model: torch.nn.Module, losses: TrainingLosses) -> None:
        """
        Helper transform gradients method that allows for injection of model.

        **NOTE**: Subclasses should implement this helper should there be a need to specialize the logic
        for transforming gradients.

        Args:
            model (torch.nn.Module): the model used to generate predictions to compute losses
            losses (TrainingLosses): The losses object from the train step
        """
        pass

    def transform_gradients(self, losses: TrainingLosses) -> None:
        """
        Hook function for model training only called after backwards pass but before optimizer step. Useful for
        transforming the gradients (such as with gradient clipping) before they are applied to the model weights.

        Args:
            losses (TrainingLosses): The losses object from the train step
        """
        return self._transform_gradients_with_model(self.model, losses)
