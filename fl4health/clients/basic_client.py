import datetime
from collections.abc import Iterator, Sequence
from logging import INFO, WARNING
from pathlib import Path
from typing import Any

import torch
from flwr.client import NumPyClient
from flwr.common.logger import log
from flwr.common.typing import Config, NDArrays, Scalar
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from fl4health.checkpointing.client_module import CheckpointMode, ClientCheckpointAndStateModule
from fl4health.metrics.base_metrics import TEST_LOSS_KEY, TEST_NUM_EXAMPLES_KEY, Metric
from fl4health.metrics.metric_managers import MetricManager
from fl4health.parameter_exchange.full_exchanger import FullParameterExchanger
from fl4health.parameter_exchange.parameter_exchanger_base import ParameterExchanger
from fl4health.reporting.base_reporter import BaseReporter
from fl4health.reporting.reports_manager import ReportsManager
from fl4health.utils.client import (
    check_if_batch_is_empty_and_verify_input,
    fold_loss_dict_into_metrics,
    maybe_progress_bar,
    move_data_to_device,
    process_and_check_validation_steps,
    set_pack_losses_with_val_metrics,
)
from fl4health.utils.config import narrow_dict_type
from fl4health.utils.early_stopper import EarlyStopper
from fl4health.utils.logging import LoggingMode
from fl4health.utils.losses import EvaluationLosses, LossMeter, LossMeterType, TrainingLosses
from fl4health.utils.random import generate_hash
from fl4health.utils.typing import LogLevel, TorchFeatureType, TorchInputType, TorchPredType, TorchTargetType


EXPECTED_OUTPUT_TUPLE_SIZE = 2


class BasicClient(NumPyClient):
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
        Base FL Client with functionality to train, evaluate, log, report and checkpoint.
        User is responsible for implementing methods: ``get_model``, ``get_optimizer``, ``get_data_loaders``,
        ``get_criterion`` Other methods can be overridden to achieve custom functionality.

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
        self.data_path = data_path
        self.device = device
        self.metrics = metrics
        self.progress_bar = progress_bar

        self.client_name = client_name if client_name is not None else generate_hash()
        log(INFO, f"Client Name: {self.client_name}")

        if checkpoint_and_state_module is not None:
            self.checkpoint_and_state_module = checkpoint_and_state_module
        else:
            # Define a default module that does nothing.
            self.checkpoint_and_state_module = ClientCheckpointAndStateModule(
                pre_aggregation=None, post_aggregation=None, state_checkpointer=None
            )

        # Initialize reporters with client information.
        self.reports_manager = ReportsManager(reporters)
        self.reports_manager.initialize(id=self.client_name, name=self.client_name)

        self.initialized = False  # Whether or not the client has been setup

        # Loss and Metric management
        self.train_loss_meter = LossMeter[TrainingLosses](loss_meter_type, TrainingLosses)
        self.val_loss_meter = LossMeter[EvaluationLosses](loss_meter_type, EvaluationLosses)
        self.train_metric_manager = MetricManager(metrics=self.metrics, metric_manager_name="train")
        self.val_metric_manager = MetricManager(metrics=self.metrics, metric_manager_name="val")
        self.test_loss_meter = LossMeter[EvaluationLosses](loss_meter_type, EvaluationLosses)
        self.test_metric_manager = MetricManager(metrics=self.metrics, metric_manager_name="test")

        # Optional variable to store the weights that the client was initialized with during each round of training
        self.initial_weights: NDArrays | None = None

        self.total_steps: int = 0  # Need to track total_steps across rounds for WANDB reporting
        self.total_epochs: int = 0  # Will remain as 0 if training by steps

        # Attributes to be initialized in setup_client
        self.parameter_exchanger: ParameterExchanger
        self.model: nn.Module
        self.optimizers: dict[str, torch.optim.Optimizer]
        self.train_loader: DataLoader
        self.val_loader: DataLoader
        self.test_loader: DataLoader | None
        self.num_train_samples: int
        self.num_val_samples: int
        self.num_test_samples: int | None
        self.learning_rate: float | None

        # User can set the early stopper for the client by instantiating the EarlyStopper class
        # and setting the patience and interval_steps attributes. The early stopper will be used to
        # stop training if the validation loss does not improve for a certain number of steps.
        self.early_stopper: EarlyStopper | None = None
        # Config can contain num_validation_steps key, which determines an upper bound
        # for the validation steps taken. If not specified, no upper bound will be enforced.
        # By specifying this in the config we cannot guarantee the validation set is the same
        # across rounds for clients.
        self.num_validation_steps: int | None = None
        # NOTE: These iterators are of type _BaseDataLoaderIter, which is not importable...so we're forced to use
        # Iterator
        self.train_iterator: Iterator | None = None
        self.val_iterator: Iterator | None = None

    def _maybe_checkpoint(self, loss: float, metrics: dict[str, Scalar], checkpoint_mode: CheckpointMode) -> None:
        """
        If checkpointer exists, maybe checkpoint model based on the provided metric values.

        Args:
            loss (float): Validation loss to potentially be used for checkpointing.
            metrics (dict[str, Scalar]): Validation metrics to potentially be used for checkpointing
            checkpoint_mode (CheckpointMode): Whether we're doing checkpointing pre- or post-aggregation on the server
                side.
        """
        self.checkpoint_and_state_module.maybe_checkpoint(self.model, loss, metrics, checkpoint_mode)

    def get_parameters(self, config: Config) -> NDArrays:
        """
        Determines which parameters are sent back to the server for aggregation. This uses a parameter exchanger to
        determine parameters sent.

        Args:
            config (Config): The config is sent by the FL server to allow for customization in the function if desired.

        Returns:
            (NDArrays): These are the parameters to be sent to the server. At minimum they represent the relevant model
                parameters to be aggregated, but can contain more information.
        """
        if not self.initialized:
            return self.setup_client_and_return_all_model_parameters(config)

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
            parameters (NDArrays): Model parameters to be injected into the client model.
            config (Config): The config is sent by the FL server to allow for customization in the function if desired.
        """
        FullParameterExchanger().pull_parameters(parameters, self.model, config)

    def setup_client_and_return_all_model_parameters(self, config: Config) -> NDArrays:
        """
        Function used to setup the client using the provided configuration and then exact all model parameters from
        ``self.model`` and return them. This function is used as a helper for ``get_parameters`` when the client
        has yet to be initialized.

        Args:
            config (Config): Configuration to be used  in setting up the client.

        Returns:
            (NDArrays): All parameters associated with the ``self.model`` property of the client.
        """
        log(INFO, "Setting up client and providing full model parameters to the server for initialization")
        if not config:
            log(
                WARNING,
                (
                    "This client has not yet been initialized and the config is empty. This may cause unexpected "
                    "failures, as setting up a client typically requires several configuration parameters, "
                    "including batch_size and current_server_round."
                ),
            )

        # If initialized is False, the server is requesting model parameters from which to initialize all other
        # clients. As such get_parameters is being called before fit or evaluate, so we must call
        # setup_client first.
        self.setup_client(config)

        # Need all parameters even if normally exchanging partial
        return FullParameterExchanger().push_parameters(self.model, config=config)

    def shutdown(self) -> None:
        """Shuts down the client. Involves shutting down W&B reporter if one exists."""
        # Shutdown reporters
        self.reports_manager.report({"shutdown": str(datetime.datetime.now())})
        self.reports_manager.shutdown()

    def process_config(self, config: Config) -> tuple[int | None, int | None, int, bool, bool]:
        """
        Method to ensure the required keys are present in config and extracts values to be returned.

        Args:
            config (Config): The config from the server.

        Returns:
            (tuple[int | None, int | None, int, bool, bool]): Returns the ``local_epochs``, ``local_steps``,
                ``current_server_round``, ``evaluate_after_fit`` and ``pack_losses_with_val_metrics``. Ensures only
                one of ``local_epochs`` and ``local_steps`` is defined in the config and sets the one that is not to
                None.

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
            (tuple[NDArrays, int, dict[str, Scalar]]): The parameters following the local training along with the
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
            (tuple[float, int, dict[str, Scalar]]): A loss associated with the evaluation, the number of samples in the
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
            (bool): Whether to perform an evaluation on the client validation set after fitting.
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
        Logs a header string. By default this is logged at the beginning of each local epoch or at the beginning of
        the round if training by steps.

        Args:
            current_round (int | None, optional): The current FL round. (Ie current server round). Defaults to None.
            current_epoch (int | None, optional): The current epoch of local training. Defaults to None.
            logging_mode (LoggingMode, optional): The logging mode to be used in logging. This mainly changes the
                way in which logging is decorated. Defaults to LoggingMode.TRAIN.
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
        Handles the logging of losses, metrics, and other information to the output file. Called only at the end of
        an epoch or server round.

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
            (tuple[str, list[tuple[LogLevel, str]]]):

                - A string to append to the header log string that typically announces the current server round and
                  current epoch at the beginning of each round or local epoch.
                - A list of tuples where the first element is a ``LogLevel`` as defined in ``fl4health.utils.``
                  typing and the second element is a string message. Each item in the list will be logged at the end of
                  each server round or epoch. Elements will also be logged at the end of validation/testing.
        """
        return "", []

    def get_client_specific_reports(self) -> dict[str, Any]:
        """
        This function can be overridden by an inheriting client to report additional client specific information to
        the ``wandb_reporter``.

        Returns:
            (dict[str, Any]): A dictionary of things to report.
        """
        return {}

    def update_metric_manager(
        self,
        preds: TorchPredType,
        target: TorchTargetType,
        metric_manager: MetricManager,
    ) -> None:
        """
        Updates a metric manager with the provided model predictions and targets. Can be overridden to modify pred and
        target inputs to the metric manager. This is useful in cases where the preds and targets needed to compute the
        loss are different than what is needed to compute metrics.

        Args:
            preds (TorchPredType): The output predictions from the model returned by ``self.predict``.
            target (TorchTargetType): The targets generated by the dataloader with which to evaluate the predictions.
            metric_manager (MetricManager): The metric manager to update.
        """
        metric_manager.update(preds, target)

    def train_step(self, input: TorchInputType, target: TorchTargetType) -> tuple[TrainingLosses, TorchPredType]:
        """
        Given a single batch of input and target data, generate predictions, compute loss, update parameters and
        optionally update metrics if they exist. (i.e. backprop on a single batch of data).
        Assumes ``self.model`` is in train mode already.

        Args:
            input (TorchInputType): The input to be fed into the model.
            target (TorchTargetType): The target corresponding to the input.

        Returns:
            (tuple[TrainingLosses, TorchPredType]): The losses object from the train step along with a dictionary of
                any predictions produced by the model.
        """
        # Clear gradients from optimizer if they exist
        self.optimizers["global"].zero_grad()

        # Call user defined methods to get predictions and compute loss
        preds, features = self.predict(input)
        target = self.transform_target(target)
        losses = self.compute_training_loss(preds, features, target)

        # Compute backward pass and update parameters with optimizer
        losses.backward["backward"].backward()
        self.transform_gradients(losses)
        self.optimizers["global"].step()

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
        # Get preds and compute loss
        with torch.no_grad():
            preds, features = self.predict(input)
            target = self.transform_target(target)
            losses = self.compute_evaluation_loss(preds, features, target)

        return losses, preds

    def train_by_epochs(
        self,
        epochs: int,
        current_round: int | None = None,
    ) -> tuple[dict[str, float], dict[str, Scalar]]:
        """
        Train locally for the specified number of epochs.

        Args:
            epochs (int): The number of epochs for local training.
            current_round (int | None, optional): The current FL round.

        Returns:
            (tuple[dict[str, float], dict[str, Scalar]]): The loss and metrics dictionary from the local training.
                Loss is a dictionary of one or more losses that represent the different components of the loss.
        """
        self.model.train()
        steps_this_round = 0  # Reset number of steps this round
        report_data: dict[str, Any] = {"round": current_round}
        continue_training = True
        for local_epoch in range(epochs):
            self.train_metric_manager.clear()
            self.train_loss_meter.clear()
            # Print initial log string on epoch start
            self._log_header_str(current_round, local_epoch)
            # update before epoch hook
            self.update_before_epoch(epoch=local_epoch)
            # Update report data dict
            report_data.update({"fit_epoch": self.total_epochs})
            for input, target in maybe_progress_bar(self.train_loader, self.progress_bar):
                self.update_before_step(steps_this_round, current_round)
                # Assume first dimension is batch size. Sampling iterators (such as Poisson batch sampling), can
                # construct empty batches. We skip the iteration if this occurs.
                if check_if_batch_is_empty_and_verify_input(input):
                    log(INFO, "Empty batch generated by data loader. Skipping step.")
                    continue

                input = move_data_to_device(input, self.device)
                target = move_data_to_device(target, self.device)
                losses, preds = self.train_step(input, target)
                self.train_loss_meter.update(losses)
                self.update_metric_manager(preds, target, self.train_metric_manager)
                self.update_after_step(steps_this_round, current_round)
                self.update_lr_schedulers(epoch=local_epoch)
                report_data.update({"fit_step_losses": losses.as_dict(), "fit_step": self.total_steps})
                report_data.update(self.get_client_specific_reports())
                self.reports_manager.report(report_data, current_round, self.total_epochs, self.total_steps)
                self.total_steps += 1
                steps_this_round += 1
                if self.early_stopper is not None and self.early_stopper.should_stop(steps_this_round):
                    log(INFO, "Early stopping criterion met. Stopping training.")
                    self.early_stopper.load_snapshot()
                    continue_training = False
                    break

            # Log and report results
            metrics = self.train_metric_manager.compute()
            loss_dict = self.train_loss_meter.compute().as_dict()
            report_data.update({"fit_epoch_metrics": metrics, "fit_epoch_losses": loss_dict})
            report_data.update(self.get_client_specific_reports())
            self.reports_manager.report(report_data, current_round, self.total_epochs)
            self._log_results(loss_dict, metrics, current_round, local_epoch)

            # Update internal epoch counter
            self.total_epochs += 1

            if not continue_training:
                break

        # Return final training metrics
        return loss_dict, metrics

    def train_by_steps(
        self,
        steps: int,
        current_round: int | None = None,
    ) -> tuple[dict[str, float], dict[str, Scalar]]:
        """
        Train locally for the specified number of steps.

        Args:
            steps (int): The number of steps to train locally.
            current_round (int | None, optional): The current FL round

        Returns:
            (tuple[dict[str, float], dict[str, Scalar]]): The loss and metrics dictionary from the local training.
                Loss is a dictionary of one or more losses that represent the different components of the loss.
        """
        self.model.train()

        # If the train_iterator hasn't been created before, we do so now.
        if self.train_iterator is None:
            # Pass loader to iterator so we can step through train loader
            self.train_iterator = iter(self.train_loader)

        self.train_loss_meter.clear()
        self.train_metric_manager.clear()
        self._log_header_str(current_round)
        report_data: dict[str, Any] = {"round": current_round}
        for step in maybe_progress_bar(range(steps), self.progress_bar):
            self.update_before_step(step, current_round)

            try:
                input, target = next(self.train_iterator)
            except StopIteration:
                # StopIteration is thrown if dataset ends. Calling iter() on the dataloader resets the loader
                # If shuffle=True for the dataloader, the data is also shuffled anew. If not, we pass through
                # the data in the same order
                self.train_iterator = iter(self.train_loader)
                input, target = next(self.train_iterator)

            # Assume first dimension is batch size. Sampling iterators (such as Poisson batch sampling), can
            # construct empty batches. We skip the iteration if this occurs.
            if check_if_batch_is_empty_and_verify_input(input):
                log(INFO, "Empty batch generated by data loader. Skipping step.")
                continue

            input = move_data_to_device(input, self.device)
            target = move_data_to_device(target, self.device)
            losses, preds = self.train_step(input, target)
            self.train_loss_meter.update(losses)
            self.update_metric_manager(preds, target, self.train_metric_manager)
            self.update_after_step(step, current_round)
            self.update_lr_schedulers(step=step)
            report_data.update({"fit_step_losses": losses.as_dict(), "fit_step": self.total_steps})
            report_data.update(self.get_client_specific_reports())
            self.reports_manager.report(report_data, current_round, None, self.total_steps)
            self.total_steps += 1
            if self.early_stopper is not None and self.early_stopper.should_stop(step):
                log(INFO, "Early stopping criterion met. Stopping training.")
                self.early_stopper.load_snapshot()
                break

        loss_dict = self.train_loss_meter.compute().as_dict()
        metrics = self.train_metric_manager.compute()

        # Log and report results
        self._log_results(loss_dict, metrics, current_round)

        return loss_dict, metrics

    def _validate_by_steps(
        self, loss_meter: LossMeter, metric_manager: MetricManager, include_losses_in_metrics: bool = False
    ) -> tuple[float, dict[str, Scalar]]:
        """
        Evaluate the model on the validation set for a fixed number of steps set by ``self.num_validation_steps``.

        Args:
            loss_meter (LossMeter): The meter to track the losses.
            metric_manager (MetricManager): The manager to track the metrics.
            include_losses_in_metrics (bool, optional): Whether or not to pack the additional losses into the metrics
                dictionary. Defaults to False.

        Returns:
            (tuple[float, dict[str, Scalar]]): The loss and a dictionary of metrics from evaluation.
        """
        assert self.num_validation_steps is not None, "num_validation_steps must be defined to use this function"

        self.model.eval()
        metric_manager.clear()
        loss_meter.clear()

        # If the val_iterator hasn't been created before, we do so now.
        if self.val_iterator is None:
            # Pass loader to iterator so we can step through validation loader
            self.val_iterator = iter(self.val_loader)

        with torch.no_grad():
            for _ in maybe_progress_bar(range(self.num_validation_steps), self.progress_bar):
                try:
                    input, target = next(self.val_iterator)
                except StopIteration:
                    # StopIteration is thrown if dataset ends. Calling iter() on the dataloader resets the loader
                    # If shuffle=True for the dataloader, the data is also shuffled anew. If not, we pass through
                    # the data in the same order
                    self.val_iterator = iter(self.val_loader)
                    input, target = next(self.val_iterator)

                input = move_data_to_device(input, self.device)
                target = move_data_to_device(target, self.device)
                losses, preds = self.val_step(input, target)
                loss_meter.update(losses)
                self.update_metric_manager(preds, target, metric_manager)

        # Compute losses and metrics over validation set
        loss_dict = loss_meter.compute().as_dict()
        metrics = metric_manager.compute()
        self._log_results(loss_dict, metrics, logging_mode=LoggingMode.VALIDATION)

        if include_losses_in_metrics:
            fold_loss_dict_into_metrics(metrics, loss_dict, LoggingMode.VALIDATION)

        return loss_dict["checkpoint"], metrics

    def _fully_validate_or_test(
        self,
        loader: DataLoader,
        loss_meter: LossMeter,
        metric_manager: MetricManager,
        logging_mode: LoggingMode = LoggingMode.VALIDATION,
        include_losses_in_metrics: bool = False,
    ) -> tuple[float, dict[str, Scalar]]:
        """
        Evaluate the model on the given validation or test dataset.

        Args:
            loader (DataLoader): The data loader for the dataset (validation or test).
            loss_meter (LossMeter): The meter to track the losses.
            metric_manager (MetricManager): The manager to track the metrics.
            logging_mode (LoggingMode, optional): The ``LoggingMode`` for whether this evaluation is for validation or
                test. Defaults to ``LoggingMode.VALIDATION``.
            include_losses_in_metrics (bool, optional): Whether or not to pack the additional losses into the metrics
                dictionary. Defaults to False.

        Returns:
            (tuple[float, dict[str, Scalar]]): The loss and a dictionary of metrics from evaluation.
        """
        assert logging_mode in [LoggingMode.VALIDATION, LoggingMode.TEST], "logging_mode must be VALIDATION or TEST"

        self.model.eval()
        metric_manager.clear()
        loss_meter.clear()
        with torch.no_grad():
            for input, target in maybe_progress_bar(loader, self.progress_bar):
                input = move_data_to_device(input, self.device)
                target = move_data_to_device(target, self.device)
                losses, preds = self.val_step(input, target)
                loss_meter.update(losses)
                self.update_metric_manager(preds, target, metric_manager)

        # Compute losses and metrics over validation set
        loss_dict = loss_meter.compute().as_dict()
        metrics = metric_manager.compute()
        self._log_results(loss_dict, metrics, logging_mode=logging_mode)

        if include_losses_in_metrics:
            fold_loss_dict_into_metrics(metrics, loss_dict, logging_mode)

        return loss_dict["checkpoint"], metrics

    def validate(self, include_losses_in_metrics: bool = False) -> tuple[float, dict[str, Scalar]]:
        """
        Validate the current model on the entire validation (or a subset thereof if ``num_validation_steps`` is not
        None) and potentially an entire test dataset if it has been defined.

        Args:
            include_losses_in_metrics (bool, optional): Determines whether to include the calculated losses into the
                metrics that are sent back to the server. Defaults to False.

        Returns:
            (tuple[float, dict[str, Scalar]]): The validation loss and a dictionary of metrics from validation
                (and test if present).
        """
        if self.num_validation_steps is None:
            val_loss, val_metrics = self._fully_validate_or_test(
                self.val_loader,
                self.val_loss_meter,
                self.val_metric_manager,
                include_losses_in_metrics=include_losses_in_metrics,
            )
        else:
            val_loss, val_metrics = self._validate_by_steps(
                self.val_loss_meter,
                self.val_metric_manager,
                include_losses_in_metrics=include_losses_in_metrics,
            )

        if self.test_loader:
            test_loss, test_metrics = self._fully_validate_or_test(
                self.test_loader,
                self.test_loss_meter,
                self.test_metric_manager,
                LoggingMode.TEST,
                include_losses_in_metrics=include_losses_in_metrics,
            )
            # There will be no clashes due to the naming convention associated with the metric managers
            if self.num_test_samples is not None:
                val_metrics[TEST_NUM_EXAMPLES_KEY] = self.num_test_samples
            val_metrics[TEST_LOSS_KEY] = test_loss
            val_metrics.update(test_metrics)

        return val_loss, val_metrics

    def get_properties(self, config: Config) -> dict[str, Scalar]:
        """
        Return properties (train and validation dataset sample counts) of client.

        Args:
            config (Config): The config from the server.

        Returns:
            (dict[str, Scalar]): A dictionary with two entries corresponding to the sample counts in
                the train and validation set.
        """
        if not self.initialized:
            self.setup_client(config)

        return {
            "num_train_samples": self.num_train_samples,
            "num_val_samples": self.num_val_samples,
        }

    def setup_client(self, config: Config) -> None:
        """
        Set dataloaders, optimizers, parameter exchangers and other attributes derived from these.
        Then set initialized attribute to True.

        Args:
            config (Config): The config from the server.
        """
        # Explicitly send the model to the desired device. This is idempotent.
        self.model = self.get_model(config).to(self.device)
        train_loader, val_loader = self.get_data_loaders(config)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = self.get_test_data_loader(config)

        self.num_validation_steps = process_and_check_validation_steps(config, self.val_loader)

        # The following lines are type ignored because torch datasets are not "Sized"
        # IE __len__ is considered optionally defined. In practice, it is almost always defined
        # and as such, we will make that assumption.
        self.num_train_samples = len(self.train_loader.dataset)  # type: ignore

        # if num_validation_steps is defined, the number of validation samples seen is
        # batch_size * num_validation_steps
        self.num_val_samples = len(self.val_loader.dataset)  # type: ignore
        if self.num_validation_steps is not None:
            assert self.val_loader.batch_size is not None, (
                "Validation batch size must be defined if we want to limit the number of validation steps"
            )
            self.num_val_samples = self.num_validation_steps * self.val_loader.batch_size

        if self.test_loader:
            self.num_test_samples = len(self.test_loader.dataset)  # type: ignore

        self.set_optimizer(config)

        # Must initialize LR scheduler after parent method initializes optimizer
        # Add lr_scheduler to dictionary if user overrides get_lr_scheduler to return
        # scheduler for given optimizer
        self.lr_schedulers = {}
        for optimizer_key in self.optimizers:
            lr_scheduler = self.get_lr_scheduler(optimizer_key, config)
            if lr_scheduler is not None:
                self.lr_schedulers[optimizer_key] = lr_scheduler

        self.criterion = self.get_criterion(config).to(self.device)
        self.parameter_exchanger = self.get_parameter_exchanger(config)

        self.reports_manager.report({"host_type": "client", "initialized": str(datetime.datetime.now())})
        self.initialized = True

    def get_parameter_exchanger(self, config: Config) -> ParameterExchanger:
        """
        Returns Full Parameter Exchangers. Subclasses that require custom Parameter Exchangers can override this.

        Args:
            config (Config): The config from server.

        Returns:
            (ParameterExchanger): Used to exchange parameters between server and client.
        """
        return FullParameterExchanger()

    def predict(self, input: TorchInputType) -> tuple[TorchPredType, TorchFeatureType]:
        """
        Computes the prediction(s), and potentially features, of the model(s) given the input.

        Args:
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
            output = self.model(input)
        elif isinstance(input, dict):
            # If input is a dictionary, then we unpack it before computing the forward pass.
            # Note that this assumes the keys of the input match (exactly) the keyword args
            # of self.model.forward().
            output = self.model(**input)
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

    def compute_loss_and_additional_losses(
        self, preds: TorchPredType, features: TorchFeatureType, target: TorchTargetType
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        """
        Computes the loss and any additional losses given predictions of the model and ground truth data.

        Args:
            preds (TorchPredType): Prediction(s) of the model(s) indexed by name.
            features (TorchFeatureType): Feature(s) of the model(s) indexed by name.
            target (TorchTargetType): Ground truth data to evaluate predictions against.

        Returns:
            (tuple[torch.Tensor, dict[str, torch.Tensor] | None]): A tuple with:

                - The tensor for the loss.
                - A dictionary of additional losses with their names and values, or None if there are no additional
                  losses.
        """
        return self.criterion(preds["prediction"], target), None

    def compute_training_loss(
        self,
        preds: TorchPredType,
        features: TorchFeatureType,
        target: TorchTargetType,
    ) -> TrainingLosses:
        """
        Computes training loss given predictions (and potentially features) of the model and ground truth data.

        Args:
            preds (TorchPredType): Prediction(s) of the model(s) indexed by name. Anything stored
                in preds will be used to compute metrics.
            features (TorchFeatureType): Feature(s) of the model(s) indexed by name.
            target (TorchTargetType): Ground truth data to evaluate predictions against.

        Returns:
            (TrainingLosses): An instance of ``TrainingLosses`` containing backward loss and additional losses
                indexed by name.
        """
        loss, additional_losses = self.compute_loss_and_additional_losses(preds, features, target)
        return TrainingLosses(backward=loss, additional_losses=additional_losses)

    def compute_evaluation_loss(
        self,
        preds: TorchPredType,
        features: TorchFeatureType,
        target: TorchTargetType,
    ) -> EvaluationLosses:
        """
        Computes evaluation loss given predictions (and potentially features) of the model and ground truth data.

        Args:
            preds (TorchPredType): Prediction(s) of the model(s) indexed by name. Anything stored
                in preds will be used to compute metrics.
            features (TorchFeatureType): Feature(s) of the model(s) indexed by name.
            target (TorchTargetType): Ground truth data to evaluate predictions against.

        Returns:
            (EvaluationLosses): An instance of ``EvaluationLosses`` containing checkpoint loss and additional losses
                indexed by name.
        """
        loss, additional_losses = self.compute_loss_and_additional_losses(preds, features, target)
        return EvaluationLosses(checkpoint=loss, additional_losses=additional_losses)

    def set_optimizer(self, config: Config) -> None:
        """
        Method called in the ``setup_client`` method to set optimizer attribute returned by used-defined
        ``get_optimizer``. In the simplest case, ``get_optimizer`` returns an optimizer. For more advanced use cases
        where a dictionary of string and optimizer are returned (i.e. APFL), the user must override this method.

        Args:
            config (Config): The config from the server.
        """
        optimizer = self.get_optimizer(config)
        assert not isinstance(optimizer, dict)
        self.optimizers = {"global": optimizer}

    def get_data_loaders(self, config: Config) -> tuple[DataLoader, DataLoader]:
        """
        User defined method that returns a PyTorch Train ``DataLoader`` and a PyTorch Validation ``DataLoader``.

        Args:
            config (Config): The config from the server.

        Returns:
            (tuple[DataLoader, DataLoader]) Tuple of length 2. The client train and validation loader.

        Raises:
            NotImplementedError: To be defined in child class.
        """
        raise NotImplementedError

    def get_test_data_loader(self, config: Config) -> DataLoader | None:
        """
        User defined method that returns a PyTorch Test DataLoader. By default, this function returns None,
        assuming that there is no test dataset to be used. If the user would like to load and evaluate a dataset,
        they need only override this function in their client class.

        Args:
            config (Config): The config from the server.

        Returns:
            (DataLoader | None): The optional client test loader.
        """
        return None

    def transform_target(self, target: TorchTargetType) -> TorchTargetType:
        """
        Method that users can extend to specify an arbitrary transformation to apply to
        the target prior to the loss being computed. Defaults to the identity transform.

        Overriding this method can be useful in a variety of scenarios such as Self Supervised
        Learning where the target is derived from the input sample itself. For example, the FedSimClr
        reference implementation overrides this method to extract features from the target, which
        is a transformed version of the input image itself.

        Args:
            target (TorchTargetType): The target or label used to compute the loss.

        Returns:
            (TorchTargetType): Identical to target.
        """
        return target

    def get_criterion(self, config: Config) -> _Loss:
        """
        User defined method that returns PyTorch loss to train model.

        Args:
            config (Config): The config from the server.

        Raises:
            NotImplementedError: To be defined in child class.
        """
        raise NotImplementedError

    def get_optimizer(self, config: Config) -> Optimizer | dict[str, Optimizer]:
        """
        Method to be defined by user that returns the PyTorch optimizer used to train models locally
        Return value can be a single torch optimizer or a dictionary of string and torch optimizer.
        Returning multiple optimizers is useful in methods like APFL which has a different optimizer
        for the local and global models.

        Args:
            config (Config): The config sent from the server.

        Returns:
            (Optimizer | dict[str, Optimizer]): An optimizer or dictionary of optimizers to train model.

        Raises:
            NotImplementedError: To be defined in child class.
        """
        raise NotImplementedError

    def get_model(self, config: Config) -> nn.Module:
        """
        User defined method that returns PyTorch model.

        Args:
            config (Config): The config from the server.

        Returns:
            (nn.Module): The client model.

        Raises:
            NotImplementedError: To be defined in child class.
        """
        raise NotImplementedError

    def get_lr_scheduler(self, optimizer_key: str, config: Config) -> LRScheduler | None:
        """
        Optional user defined method that returns learning rate scheduler
        to be used throughout training for the given optimizer. Defaults to None.

        Args:
            optimizer_key (str): The key in the optimizer dict corresponding
                to the optimizer we are optionally defining a learning rate
                scheduler for.
            config (Config): The config from the server.

        Returns:
            (LRScheduler | None): Client learning rate schedulers.
        """
        return None

    def update_lr_schedulers(self, step: int | None = None, epoch: int | None = None) -> None:
        """
        Updates any schedulers that exist. Can be overridden to customize update logic based on client state
        (i.e ``self.total_steps``).

        Args:
            step (int | None): If using ``local_steps``, current step of this round. Otherwise None.
            epoch (int | None): If using ``local_epochs`` current epoch of this round. Otherwise None.
        """
        assert (step is None) ^ (epoch is None)

        for lr_scheduler in self.lr_schedulers.values():
            lr_scheduler.step()  # Update LR

    def update_before_train(self, current_server_round: int) -> None:
        """
        Hook method called before training with the number of current server rounds performed.

        **NOTE**: This method is called immediately **AFTER** the aggregated parameters are received from the server.
        For example, used by MOON and FENDA to save global modules after aggregation.

        Args:
            current_server_round (int): The number of current server round.
        """
        pass

    def update_after_train(self, local_steps: int, loss_dict: dict[str, float], config: Config) -> None:
        """
        Hook method called after training with the number of ``local_steps`` performed over the FL round and
        the corresponding loss dictionary. For example, used by Scaffold to update the control variates
        after a local round of training. Also used by FedProx to update the current loss based on the loss
        returned during training. Also used by MOON and FENDA to save trained modules weights before
        aggregation.

        Args:
            local_steps (int): The number of steps so far in the round in the local training.
            loss_dict (dict[str, float]): A dictionary of losses from local training.
            config (Config): The config from the server
        """
        pass

    def update_before_step(self, step: int, current_round: int | None = None) -> None:
        """
        Hook method called before local train step.

        Args:
            step (int): The local training step that was most recently completed. Resets only at the end of the round.
            current_round (int | None, optional): The current FL server round.
        """
        pass

    def update_after_step(self, step: int, current_round: int | None = None) -> None:
        """
        Hook method called after local train step on client. Step is an integer that represents the local training
        step that was most recently completed. For example, used by the APFL method to update the alpha value after a
        training a step. Also used by the MOON, FENDA and Ditto to update optimized beta value for MK-MMD loss after
        n steps.

        Args:
            step (int): The step number in local training that was most recently completed. Resets only at the end of
                the round.
            current_round (int | None, optional): The current FL server round.
        """
        pass

    def update_before_epoch(self, epoch: int) -> None:
        """
        Hook method called before local epoch on client. Only called if client is being trained by epochs
        (i.e. using ``local_epochs`` key instead of local steps in the server config file).

        Args:
            epoch (int): Integer representing the epoch about to begin
        """
        pass

    def transform_gradients(self, losses: TrainingLosses) -> None:
        """
        Hook function for model training only called after backwards pass but before optimizer step. Useful for
        transforming the gradients (such as with gradient clipping) before they are applied to the model weights.

        Args:
            losses (TrainingLosses): The losses object from the train step
        """
        pass

    def _save_client_state(self) -> None:
        """
        Save a checkpoint of the client's state as defined by the state_checkpointer's snapshot_attrs.
        By default, snapshot_attrs includes attributes such as client name, total steps, lr schedulers,
        metrics reporter, and optimizer states. You can override snapshot_attrs in the state_checkpointer to
        customize which attributes are saved in the checkpoint.
        """
        assert self.checkpoint_and_state_module.state_checkpointer is not None
        self.checkpoint_and_state_module.save_state(self)

    def _load_client_state(self) -> bool:
        """
        Load checkpoint dict consisting of client name, total steps, lr schedulers, metrics reporter and optimizers
        state. Method can be overridden to augment loaded checkpointed state.
        """
        assert self.checkpoint_and_state_module.state_checkpointer is not None
        log(INFO, "Loading client state from checkpoint")
        return self.checkpoint_and_state_module.maybe_load_state(self)
