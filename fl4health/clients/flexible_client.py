import datetime
import warnings
from collections.abc import Sequence
from logging import INFO, WARN
from pathlib import Path
from typing import Any

import torch
from flwr.common.logger import log
from flwr.common.typing import Config, NDArrays, Scalar
from torch import nn
from torch.optim import Optimizer

from fl4health.checkpointing.client_module import CheckpointMode, ClientCheckpointAndStateModule
from fl4health.clients.basic_client import BasicClient
from fl4health.metrics.base_metrics import Metric
from fl4health.metrics.metric_managers import MetricManager
from fl4health.parameter_exchange.full_exchanger import FullParameterExchanger
from fl4health.reporting.base_reporter import BaseReporter
from fl4health.utils.client import (
    set_pack_losses_with_val_metrics,
)
from fl4health.utils.config import narrow_dict_type
from fl4health.utils.logging import LoggingMode
from fl4health.utils.losses import EvaluationLosses, LossMeterType, TrainingLosses
from fl4health.utils.typing import LogLevel, TorchFeatureType, TorchInputType, TorchPredType, TorchTargetType


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
        User is responsible for implementing methods: ``get_model``, ``get_optimizer``, ``get_data_loaders``,
        ``get_criterion`` Other methods can be overridden to achieve custom functionality.

        Args:
            data_path (Path): path to the data to be used to load the data for client-side training
            metrics (Sequence[Metric]): Metrics to be computed based on the labels and predictions of the client model
            device (torch.device): Device indicator for where to send the model, batches, labels etc. Often "cpu" or
                "cuda"
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

    def _maybe_checkpoint(self, loss: float, metrics: dict[str, Scalar], checkpoint_mode: CheckpointMode) -> None:
        """
        If checkpointer exists, maybe checkpoint model based on the provided metric values.

        Args:
            loss (float): validation loss to potentially be used for checkpointing
            metrics (dict[str, float]): validation metrics to potentially be used for checkpointing
        """
        self.checkpoint_and_state_module.maybe_checkpoint(self.model, loss, metrics, checkpoint_mode)

    def get_parameters(self, config: Config) -> NDArrays:
        """
        Determines which parameters are sent back to the server for aggregation. This uses a parameter exchanger to
        determine parameters sent.

        Args:
            config (Config): The config is sent by the FL server to allow for customization in the function if desired.

        Returns:
            NDArrays: These are the parameters to be sent to the server. At minimum they represent the relevant model
            parameters to be aggregated, but can contain more information.
        """
        if not self.initialized:
            log(
                INFO,
                "Setting up client and providing full model parameters to the server for initialization",
            )

            # If initialized==False, the server is requesting model parameters from which to initialize all other
            # clients. As such get_parameters is being called before fit or evaluate, so we must call
            # setup_client first.
            self.setup_client(config)

            # Need all parameters even if normally exchanging partial
            return FullParameterExchanger().push_parameters(self.model, config=config)
        assert self.model is not None and self.parameter_exchanger is not None
        # If the client has early stopping module and the patience is None, we load the best saved state
        # to send the best checkpointed local model's parameters to the server
        self._maybe_load_saved_best_local_model_state()
        return self.parameter_exchanger.push_parameters(self.model, config=config)

    def _maybe_load_saved_best_local_model_state(self) -> None:
        if self.early_stopper is not None and self.early_stopper.patience is None:
            log(INFO, "Loading saved best model's state before sending model to server.")
            self.early_stopper.load_snapshot(["model"])

    def set_parameters(self, parameters: NDArrays, config: Config, fitting_round: bool) -> None:
        """
        Sets the local model parameters transferred from the server using a parameter exchanger to coordinate how
        parameters are set. In the first fitting round, we assume the full model is being
        initialized and use the ``FullParameterExchanger()`` to set all model weights.
        Otherwise, we use the appropriate parameter exchanger defined by the user depending on the
        federated learning algorithm being used.

        Args:
            parameters (NDArrays): Parameters have information about model state to be added to the relevant client
                model but may contain more information than that.
            config (Config): The config is sent by the FL server to allow for customization in the function if desired.
            fitting_round (bool): Boolean that indicates whether the current federated learning round is a
                fitting round or an evaluation round.
                This is used to help determine which parameter exchange should be used for pulling parameters.
                A full parameter exchanger is only used if the current federated learning round is the very
                first fitting round.
        """
        assert self.model is not None
        current_server_round = narrow_dict_type(config, "current_server_round", int)
        if current_server_round == 1 and fitting_round:
            self.initialize_all_model_weights(parameters, config)
        else:
            assert self.parameter_exchanger is not None
            self.parameter_exchanger.pull_parameters(parameters, self.model, config)

    def initialize_all_model_weights(self, parameters: NDArrays, config: Config) -> None:
        """
        If this is the first time we're initializing the model weights, we use the ``FullParameterExchanger`` to
        initialize all model components. Subclasses that require custom model initialization can override this.

        Args:
            parameters (NDArrays): Model parameters to be injected into the client model
            config (Config): The config is sent by the FL server to allow for customization in the function if desired.
        """
        FullParameterExchanger().pull_parameters(parameters, self.model, config)

    def shutdown(self) -> None:
        """
        Shuts down the client.

        NOTE: Involves shutting down W&B reporter if one exists.
        """
        # Shutdown reporters
        self.reports_manager.report({"shutdown": str(datetime.datetime.now())})
        self.reports_manager.shutdown()

    def process_config(self, config: Config) -> tuple[int | None, int | None, int, bool, bool]:
        """
        Method to ensure the required keys are present in config and extracts values to be returned.

        Args:
            config (Config): The config from the server.

        Returns:
            tuple[int | None, int | None, int, bool, bool]: Returns the ``local_epochs``, ``local_steps``,
            ``current_server_round``, ``evaluate_after_fit`` and ``pack_losses_with_val_metrics``. Ensures only one of
            ``local_epochs`` and ``local_steps`` is defined in the config and sets the one that is not to None.

        Raises:
            ValueError: If the config contains both ``local_steps`` and local epochs or if ``local_steps``,
                ``local_epochs`` or ``current_server_round`` is of the wrong type (int).
        """
        current_server_round = narrow_dict_type(config, "current_server_round", int)

        # Parse config to determine train by steps or train by epochs
        if ("local_epochs" in config) and ("local_steps" in config):
            raise ValueError("Config cannot contain both local_epochs and local_steps. Please specify only one.")
        if "local_epochs" in config:
            local_epochs = narrow_dict_type(config, "local_epochs", int)
            local_steps = None
        elif "local_steps" in config:
            local_steps = narrow_dict_type(config, "local_steps", int)
            local_epochs = None
        else:
            raise ValueError("Must specify either local_epochs or local_steps in the Config.")

        try:
            evaluate_after_fit = narrow_dict_type(config, "evaluate_after_fit", bool)
        except ValueError:
            evaluate_after_fit = False

        pack_losses_with_val_metrics = set_pack_losses_with_val_metrics(config)

        # Either local epochs or local steps is none based on what key is passed in the config
        return local_epochs, local_steps, current_server_round, evaluate_after_fit, pack_losses_with_val_metrics

    def fit(self, parameters: NDArrays, config: Config) -> tuple[NDArrays, int, dict[str, Scalar]]:
        """
        Processes config, initializes client (if first round) and performs training based on the passed config.
        If ``per_round_checkpointer`` is not None, on initialization the client checks if a checkpointed client state
        exists to load and at the end of each round the client state is saved.

        Args:
            parameters (NDArrays): The parameters of the model to be used in fit.
            config (NDArrays): The config from the server.

        Returns:
            tuple[NDArrays, int, dict[str, Scalar]]: The parameters following the local training along with the
            number of samples in the local training dataset and the computed metrics throughout the fit.

        Raises:
            ValueError: If ``local_steps`` or ``local_epochs`` is not specified in config.
        """
        round_start_time = datetime.datetime.now()
        local_epochs, local_steps, current_server_round, evaluate_after_fit, pack_losses_with_val_metrics = (
            self.process_config(config)
        )

        if not self.initialized:
            self.setup_client(config)

            if self.checkpoint_and_state_module.state_checkpointer is not None:
                # If this is the first time the client is being setup, we also attempt to load any existing state
                # If no state exists, we assume this is a fresh run. State is useful, for example, in restarting FL
                # training that was interrupted or failed part way through.
                state_load_success = self._load_client_state()
                if state_load_success:
                    log(INFO, "Successfully loaded client state.")
                else:
                    log(INFO, "Client state was not loaded.")

        self.set_parameters(parameters, config, fitting_round=True)

        self.update_before_train(current_server_round)

        fit_start_time = datetime.datetime.now()
        if local_epochs is not None:
            loss_dict, metrics = self.train_by_epochs(local_epochs, current_server_round)
            local_steps = len(self.train_loader) * local_epochs  # total steps over training round
        elif local_steps is not None:
            loss_dict, metrics = self.train_by_steps(local_steps, current_server_round)
        else:
            raise ValueError("Must specify either local_epochs or local_steps in the Config.")
        fit_end_time = datetime.datetime.now()

        # Perform necessary updates after training has completed for the current FL round
        self.update_after_train(local_steps, loss_dict, config)

        # Check if we should run an evaluation with validation data after fit
        # (for example, this is used by FedDGGA)
        if self._should_evaluate_after_fit(evaluate_after_fit):
            validation_loss, validation_metrics = self.validate(pack_losses_with_val_metrics)
            metrics.update(validation_metrics)
            # We perform a pre-aggregation checkpoint if applicable
            self._maybe_checkpoint(validation_loss, validation_metrics, CheckpointMode.PRE_AGGREGATION)

        # Notes on report values:
        #   - Train by steps: round metrics/losses are computed using all samples from the round
        #   - Train by epochs: round metrics/losses computed using only the samples from the final epoch of the round
        #   - fit_round_metrics: Computed at the end of the round on the samples directly
        #   - fit_round_losses: The average of the losses computed for each step.
        #       * (Hence likely higher than the final loss of the round.)
        self.reports_manager.report(
            {
                "fit_round_metrics": metrics,
                "fit_round_losses": loss_dict,
                "round": current_server_round,
                "round_start": str(round_start_time),
                "round_end": str(datetime.datetime.now()),
                "fit_round_start": str(fit_start_time),
                "fit_round_time_elapsed": round((fit_end_time - fit_start_time).total_seconds()),
                "fit_round_end": str(fit_end_time),
                "fit_step": self.total_steps,
                "fit_epoch": self.total_epochs,
            },
            current_server_round,
        )

        # After local client training has finished, checkpoint client state if a state checkpointer is defined
        if self.checkpoint_and_state_module.state_checkpointer is not None:
            self._save_client_state()

        # FitRes should contain local parameters, number of examples on client, and a dictionary holding metrics
        # calculation results.
        return (
            self.get_parameters(config),
            self.num_train_samples,
            metrics,
        )

    def evaluate(self, parameters: NDArrays, config: Config) -> tuple[float, int, dict[str, Scalar]]:
        """
        Evaluates the model on the validation set, and test set (if defined).

        Args:
            parameters (NDArrays): The parameters of the model to be evaluated.
            config (NDArrays): The config object from the server.

        Returns:
            tuple[float, int, dict[str, Scalar]]: A loss associated with the evaluation, the number of samples in the
            validation/test set and the ``metric_values`` associated with evaluation.
        """
        if not self.initialized:
            self.setup_client(config)

        start_time = datetime.datetime.now()
        current_server_round = narrow_dict_type(config, "current_server_round", int)

        pack_losses_with_val_metrics = set_pack_losses_with_val_metrics(config)

        self.set_parameters(parameters, config, fitting_round=False)
        loss, metrics = self.validate(pack_losses_with_val_metrics)
        end_time = datetime.datetime.now()
        elapsed = end_time - start_time

        # Checkpoint based on the loss and metrics produced during validation AFTER server-side aggregation
        # NOTE: This assumes that the loss returned in the checkpointing loss
        self._maybe_checkpoint(loss, metrics, CheckpointMode.POST_AGGREGATION)

        self.reports_manager.report(
            {
                "eval_round_metrics": metrics,
                "eval_round_loss": loss,
                "eval_round_start": str(start_time),
                "eval_round_time_elapsed": round(elapsed.total_seconds()),
                "eval_round_end": str(end_time),
                "fit_step": self.total_steps,
                "fit_epoch": self.total_epochs,
                "round": current_server_round,
            },
            current_server_round,
        )

        # EvaluateRes should return the loss, number of examples on client, and a dictionary holding metrics
        # calculation results.
        return (
            loss,
            self.num_val_samples,
            metrics,
        )

    def _should_evaluate_after_fit(self, evaluate_after_fit: bool) -> bool:
        """
        Function to determine whether to trigger an evaluation of the model on the validation set immediately after
        completing the local training round. The user can request this explicitly by setting evaluate_after_fit to
        true in the config, or implicitly by specifying a pre-aggregation checkpoint module.

        Args:
            evaluate_after_fit (bool): Whether the user explicitly specified that they would like an evaluate after
                fit to be performed through the config.

        Returns:
            bool: Whether to perform an evaluation on the client validation set after fitting.
        """
        pre_aggregation_checkpointing_enabled = (
            self.checkpoint_and_state_module is not None
            and self.checkpoint_and_state_module.pre_aggregation is not None
        )
        return evaluate_after_fit or pre_aggregation_checkpointing_enabled

    def _log_header_str(
        self,
        current_round: int | None = None,
        current_epoch: int | None = None,
        logging_mode: LoggingMode = LoggingMode.TRAIN,
    ) -> None:
        """
        Logs a header string.

        NOTE: By default this is logged at the beginning of each local
        epoch or at the beginning of the round if training by steps

        Args:
            current_round (int | None, optional): The current FL round. (Ie current
                server round). Defaults to None.
            current_epoch (int | None, optional): The current epoch of local
                training. Defaults to None.
        """
        log_str = f"Current FL Round: {int(current_round)} " if current_round is not None else ""
        log_str += f"Current Epoch: {int(current_epoch)} " if current_epoch is not None else ""

        # Maybe add client specific info to initial log string
        client_str, _ = self.get_client_specific_logs(current_round, current_epoch, logging_mode)

        log_str += client_str

        log(INFO, "")  # For aesthetics
        log(INFO, log_str)

    def _log_results(
        self,
        loss_dict: dict[str, float],
        metrics_dict: dict[str, Scalar],
        current_round: int | None = None,
        current_epoch: int | None = None,
        logging_mode: LoggingMode = LoggingMode.TRAIN,
    ) -> None:
        """
        Handles the logging of losses, metrics, and other information to the output file.

        NOTE: Called only at the end of an epoch or server round

        Args:
            loss_dict (dict[str, float]): A dictionary of losses to log.
            metrics_dict (dict[str, Scalar]): A dictionary of the metric to log.
            current_round (int | None): The current FL round (i.e., current server round).
            current_epoch (int | None): The current epoch of local training.
            logging_mode (LoggingMode): The logging mode (Training, Validation, or Testing).
        """
        _, client_logs = self.get_client_specific_logs(current_round, current_epoch, logging_mode)

        # Get Metric Prefix
        metric_prefix = logging_mode.value

        # Log losses if any were provided
        if len(loss_dict.keys()) > 0:
            log(INFO, f"Client {metric_prefix} Losses:")
            [log(INFO, f"\t {key}: {str(val)}") for key, val in loss_dict.items()]

        # Log metrics if any
        if len(metrics_dict.keys()) > 0:
            log(INFO, f"Client {metric_prefix} Metrics:")
            [log(INFO, f"\t {key}: {str(val)}") for key, val in metrics_dict.items()]

        # Add additional logs specific to client
        if len(client_logs) > 0:
            [log(level.value, msg) for level, msg in client_logs]

    def get_client_specific_logs(
        self,
        current_round: int | None,
        current_epoch: int | None,
        logging_mode: LoggingMode,
    ) -> tuple[str, list[tuple[LogLevel, str]]]:
        """
        This function can be overridden to provide any client specific information to the basic client logging.
        For example, perhaps a client uses an LR scheduler and wants the LR to be logged each epoch. Called at the
        beginning and end of each server round or local epoch. Also called at the end of validation/testing.

        Args:
            current_round (int | None): The current FL round (i.e., current server round).
            current_epoch (int | None): The current epoch of local training.
            logging_mode (LoggingMode): The logging mode (Training, Validation, or Testing).

        Returns:
            tuple[str, list[tuple[LogLevel, str]]]:

            - A string to append to the header log string that typically announces the current server round and
              current epoch at the beginning of each round or local epoch.
            - A list of tuples where the first element is a LogLevel as defined in ``fl4health.utils.``
              typing and the second element is a string message. Each item in the list will be logged at the end of
              each server round or epoch. Elements will also be logged at the end of validation/testing.
        """
        return "", []

    def get_client_specific_reports(self) -> dict[str, Any]:
        """
        Get client specific reports.

        NOTE: This function can be overridden by an inheriting client to report
        additional client specific information to the ``wandb_reporter``

        Returns:
            dict[str, Any]: A dictionary of things to report
        """
        return {}

    def update_metric_manager(
        self,
        preds: TorchPredType,
        target: TorchTargetType,
        metric_manager: MetricManager,
    ) -> None:
        """
        Updates a metric manager with the provided model predictions and
        targets. Can be overridden to modify pred and target inputs to the
        metric manager. This is useful in cases where the preds and targets
        needed to compute the loss are different than what is needed to compute
        metrics.

        Args:
            preds (TorchPredType): The output predictions from the model
                returned by self.predict
            target (TorchTargetType): The targets generated by the dataloader to
                to evaluate the predictions with
            metric_manager (MetricManager): The metric manager to update
        """
        metric_manager.update(preds, target)

    def _compute_preds_and_losses(
        self, model: nn.Module, optimizer: Optimizer, input: TorchInputType, target: TorchTargetType
    ) -> tuple[TrainingLosses, TorchPredType]:
        """
        Helper method within the train step for computing preds and losses.

        NOTE: Subclasses should implement this helper method if there is a need
        to specialize this part of the overall train step.

        Args:
            model (nn.Module): the model used to make predictions
            optimizer (Optimizer): the associated optimizer
            input (TorchInputType): The input to be fed into the model.
            target (TorchTargetType): The target corresponding to the input.

        Returns:
            tuple[TrainingLosses, TorchPredType]: The losses object from the train step along with
            a dictionary of any predictions produced by the model prior to the
            application of the backwards phase
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

        NOTE: Subclasses should implement this helper method if there is a need
        to specialize this part of the overall train step.

        Args:
            model (nn.Module): the model used for making predictions. Passed here in case subclasses need it.
            optimizer (Optimizer): the optimizer with which we take the step
            losses (TrainingLosses): the losses to apply backwards on

        Returns:
            TrainingLosses: The losses object post backwards application
        """
        # Compute backward pass and update parameters with optimizer
        losses.backward["backward"].backward()
        self.transform_gradients(losses)
        optimizer.step()

        return losses

    def _train_step_with_model_and_optimizer(
        self, model: torch.nn.Module, optimizer: Optimizer, input: TorchInputType, target: TorchTargetType
    ) -> tuple[TrainingLosses, TorchPredType]:
        """
        Helper train step method that allows for injection of model and optimizer.

        NOTE: Subclasses should implement this method if there is a need to specialize
        the train_step logic.

        Args:
            input (TorchInputType): The input to be fed into the model.
            target (TorchTargetType): The target corresponding to the input.

        Returns:
            tuple[TrainingLosses, TorchPredType]: The losses object from the train step along with
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
            tuple[TrainingLosses, TorchPredType]: The losses object from the train step along with
            a dictionary of any predictions produced by the model.
        """
        return self._train_step_with_model_and_optimizer(self.model, self.optimizers["global"], input, target)

    def _val_step_with_model(
        self, model: nn.Module, input: TorchInputType, target: TorchTargetType
    ) -> tuple[EvaluationLosses, TorchPredType]:
        """
        Helper method for val_step that allows for injection of model.

        NOTE: Subclasses should implement this method if there is a need to
        specialize the val_step logic.

        Args:
            input (TorchInputType): The input to be fed into the model.
            target (TorchTargetType): The target corresponding to the input.

        Returns:
            tuple[EvaluationLosses, TorchPredType]: The losses object from the val step along with a dictionary of the
            predictions produced by the model.
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
            tuple[EvaluationLosses, TorchPredType]: The losses object from the val step along with a dictionary of the
            predictions produced by the model.
        """
        return self._val_step_with_model(self.model, input, target)

    def predict_with_model(
        self, model: torch.nn.Module, input: TorchInputType
    ) -> tuple[TorchPredType, TorchFeatureType]:
        """
        Helper predict method that allows for injection of model.

        NOTE: Subclasses should implement this method if there is need to specialize
        the predict logic of the client.

        Args:
            model (torch.nn.Module): the model with which to make predictions
            input (TorchInputType): Inputs to be fed into the model. If input is of type ``dict[str, torch.Tensor]``,
                it is assumed that the keys of input match the names of the keyword arguments of
                ``self.model.forward().`

        Returns:
            tuple[TorchPredType, TorchFeatureType]: A tuple in which the first element contains a dictionary of
            predictions indexed by name and the second element contains intermediate activations indexed by name. By
            passing features, we can compute losses such as the contrastive loss in MOON. All predictions included in
            dictionary will by default be used to compute metrics separately.

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
            if len(output) != 2:
                raise ValueError(f"Output tuple should have length 2 but has length {len(output)}")
            preds, features = output
            return preds, features
        raise ValueError("Model forward did not return a tensor, dictionary of tensors, or tuple of tensors")
