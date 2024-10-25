import copy
import datetime
from collections.abc import Iterable, Sequence
from enum import Enum
from logging import INFO, WARNING
from pathlib import Path
from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn
from flwr.client import NumPyClient
from flwr.common.logger import LOG_COLORS, log
from flwr.common.typing import Config, NDArrays, Scalar
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from fl4health.checkpointing.checkpointer import PerRoundCheckpointer
from fl4health.checkpointing.client_module import CheckpointMode, ClientCheckpointModule
from fl4health.parameter_exchange.full_exchanger import FullParameterExchanger
from fl4health.parameter_exchange.parameter_exchanger_base import ParameterExchanger
from fl4health.reporting.base_reporter import BaseReporter
from fl4health.reporting.reports_manager import ReportsManager
from fl4health.utils.config import narrow_dict_type, narrow_dict_type_and_set_attribute
from fl4health.utils.losses import EvaluationLosses, LossMeter, LossMeterType, TrainingLosses
from fl4health.utils.metrics import TEST_LOSS_KEY, TEST_NUM_EXAMPLES_KEY, Metric, MetricManager
from fl4health.utils.random import generate_hash
from fl4health.utils.typing import LogLevel, TorchFeatureType, TorchInputType, TorchPredType, TorchTargetType


class LoggingMode(Enum):
    TRAIN = "Training"
    VALIDATION = "Validation"
    TEST = "Testing"


class BasicClient(NumPyClient):
    def __init__(
        self,
        data_path: Path,
        metrics: Sequence[Metric],
        device: torch.device,
        loss_meter_type: LossMeterType = LossMeterType.AVERAGE,
        checkpointer: Optional[ClientCheckpointModule] = None,
        reporters: Sequence[BaseReporter] | None = None,
        progress_bar: bool = False,
        intermediate_client_state_dir: Optional[Path] = None,
        client_name: Optional[str] = None,
    ) -> None:
        """
        Base FL Client with functionality to train, evaluate, log, report and checkpoint.
        User is responsible for implementing methods: get_model, get_optimizer, get_data_loaders, get_criterion
        Other methods can be overridden to achieve custom functionality.

        Args:
            data_path (Path): path to the data to be used to load the data for client-side training
            metrics (Sequence[Metric]): Metrics to be computed based on the labels and predictions of the client model
            device (torch.device): Device indicator for where to send the model, batches, labels etc. Often 'cpu' or
                'cuda'
            loss_meter_type (LossMeterType, optional): Type of meter used to track and compute the losses over
                each batch. Defaults to LossMeterType.AVERAGE.
            checkpointer (Optional[ClientCheckpointModule], optional): Checkpointer module defining when and how to
                do checkpointing during client-side training. No checkpointing is done if not provided. Defaults to
                None.
            reporters (Sequence[BaseReporter], optional): A sequence of FL4Health
                reporters which the client should send data to.
            progress_bar (bool): Whether or not to display a progress bar
                during client training and validation. Uses tqdm. Defaults to
                False
            intermediate_client_state_dir (Optional[Path]): An optional path to store per round
                checkpoints.
            client_name (str): An optional client name that uniquely identifies a client.
                If not passed, a hash is randomly generated.
        """

        self.data_path = data_path
        self.device = device
        self.metrics = metrics
        self.checkpointer = checkpointer
        self.progress_bar = progress_bar
        self.client_name = client_name if client_name is not None else generate_hash()

        self.per_round_checkpointer: Union[None, PerRoundCheckpointer]

        if intermediate_client_state_dir is not None:
            log(
                WARNING,
                "intermediate_client_state_dir is not None. Creating PerRoundCheckpointer. \
                This functionality still experimental and only supported for \
                FlServerWithCheckpointing and NnunetServer currently.",
            )
            self.per_round_checkpointer = PerRoundCheckpointer(
                intermediate_client_state_dir, Path(f"client_{self.client_name}.pt")
            )
        else:
            self.per_round_checkpointer = None

        # Initialize reporters with client information.
        self.reports_manager = ReportsManager(reporters)
        self.reports_manager.initialize(id=self.client_name)

        self.initialized = False  # Whether or not the client has been setup

        # Loss and Metric management
        self.train_loss_meter = LossMeter[TrainingLosses](loss_meter_type, TrainingLosses)
        self.val_loss_meter = LossMeter[EvaluationLosses](loss_meter_type, EvaluationLosses)
        self.train_metric_manager = MetricManager(metrics=self.metrics, metric_manager_name="train")
        self.val_metric_manager = MetricManager(metrics=self.metrics, metric_manager_name="val")
        self.test_loss_meter = LossMeter[EvaluationLosses](loss_meter_type, EvaluationLosses)
        self.test_metric_manager = MetricManager(metrics=self.metrics, metric_manager_name="test")

        # Optional variable to store the weights that the client was initialized with during each round of training
        self.initial_weights: Optional[NDArrays] = None

        self.total_steps: int = 0  # Need to track total_steps across rounds for WANDB reporting

        # Attributes to be initialized in setup_client
        self.parameter_exchanger: ParameterExchanger
        self.model: nn.Module
        self.optimizers: dict[str, torch.optim.Optimizer]
        self.train_loader: DataLoader
        self.val_loader: DataLoader
        self.test_loader: Optional[DataLoader]
        self.num_train_samples: int
        self.num_val_samples: int
        self.num_test_samples: Optional[int] = None
        self.learning_rate: Optional[float] = None

    def _maybe_checkpoint(self, loss: float, metrics: dict[str, Scalar], checkpoint_mode: CheckpointMode) -> None:
        """
        If checkpointer exists, maybe checkpoint model based on the provided metric values.

        Args:
            loss (float): validation loss to potentially be used for checkpointing
            metrics (dict[str, float]): validation metrics to potentially be used for checkpointing
        """
        if self.checkpointer:
            self.checkpointer.maybe_checkpoint(self.model, loss, metrics, checkpoint_mode)

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
        else:
            assert self.model is not None and self.parameter_exchanger is not None
            return self.parameter_exchanger.push_parameters(self.model, config=config)

    def set_parameters(self, parameters: NDArrays, config: Config, fitting_round: bool) -> None:
        """
        Sets the local model parameters transferred from the server using a parameter exchanger to coordinate how
        parameters are set. In the first fitting round, we assume the full model is being
        initialized and use the FullParameterExchanger() to set all model weights.
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
        If this is the first time we're initializing the model weights, we use the FullParameterExchanger to
        initialize all model components.
        Subclasses that require custom model initialization can override this.

        Args:
            parameters (NDArrays): Model parameters to be injected into the client model
            config (Config): The config is sent by the FL server to allow for customization in the function if desired.
        """
        FullParameterExchanger().pull_parameters(parameters, self.model, config)

    def shutdown(self) -> None:
        """
        Shuts down the client. Involves shutting down W&B reporter if one exists.
        """
        # Shutdown reporters
        self.reports_manager.report({"shutdown": str(datetime.datetime.now())})
        self.reports_manager.shutdown()

    def process_config(self, config: Config) -> Tuple[Union[int, None], Union[int, None], int, bool]:
        """
        Method to ensure the required keys are present in config and extracts values to be returned.

        Args:
            config (Config): The config from the server.

        Returns:
            Tuple[Union[int, None], Union[int, None], int, bool]: Returns the local_epochs, local_steps,
                current_server_round and evaluate_after_fit. Ensures only one of local_epochs and local_steps
                is defined in the config and sets the one that is not to None.

        Raises:
            ValueError: If the config contains both local_steps and local epochs or if local_steps, local_epochs or
                current_server_round is of the wrong type (int).
        """
        current_server_round = narrow_dict_type(config, "current_server_round", int)

        if ("local_epochs" in config) and ("local_steps" in config):
            raise ValueError("Config cannot contain both local_epochs and local_steps. Please specify only one.")
        elif "local_epochs" in config:
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

        # Either local epochs or local steps is none based on what key is passed in the config
        return local_epochs, local_steps, current_server_round, evaluate_after_fit

    def fit(self, parameters: NDArrays, config: Config) -> Tuple[NDArrays, int, dict[str, Scalar]]:
        """
        Processes config, initializes client (if first round) and performs training based on the passed config.
        If per_round_checkpointer is not None, on initialization the client checks if a checkpointed client state
        exists to load and at the end of each round the client state is saved.

        Args:
            parameters (NDArrays): The parameters of the model to be used in fit.
            config (NDArrays): The config from the server.

        Returns:
            Tuple[NDArrays, int, dict[str, Scalar]]: The parameters following the local training along with the
            number of samples in the local training dataset and the computed metrics throughout the fit.

        Raises:
            ValueError: If local_steps or local_epochs is not specified in config.
        """
        round_start_time = datetime.datetime.now()
        local_epochs, local_steps, current_server_round, evaluate_after_fit = self.process_config(config)

        if not self.initialized:
            self.setup_client(config)

            # If per_round_checkpointer not None and checkpoint exists load it and set attributes.
            # Model not updated because FL restarted from most recent FL round (redo preempted round)
            if self.per_round_checkpointer is not None and self.per_round_checkpointer.checkpoint_exists():
                self.load_client_state()

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
            validation_loss, validation_metrics = self.evaluate_after_fit()
            metrics.update(validation_metrics)
            # We perform a pre-aggregation checkpoint if applicable
            self._maybe_checkpoint(validation_loss, validation_metrics, CheckpointMode.PRE_AGGREGATION)

        self.reports_manager.report(
            {
                "fit_metrics": metrics,
                "fit_losses": loss_dict,
                "round": current_server_round,
                "round_start": str(round_start_time),
                "round_end": str(datetime.datetime.now()),
                "fit_start": str(fit_start_time),
                "fit_end": str(fit_end_time),
            },
            current_server_round,
        )

        # After local client training has finished, checkpoint client state
        # if per_round_checkpointer is not None
        if self.per_round_checkpointer is not None:
            self.save_client_state()

        # FitRes should contain local parameters, number of examples on client, and a dictionary holding metrics
        # calculation results.
        return (
            self.get_parameters(config),
            self.num_train_samples,
            metrics,
        )

    def evaluate_after_fit(self) -> Tuple[float, dict[str, Scalar]]:
        """
        Run self.validate right after fit to collect metrics on the local model against validation data.

        Returns: (dict[str, Scalar]) a dictionary with the metrics.

        """
        loss, metric_values = self.validate()
        # The computed loss value is packed into the metrics dictionary, perhaps for use on the server-side
        metrics_after_fit = {
            **metric_values,  # type: ignore
            "val - loss": loss,
        }
        return loss, metrics_after_fit

    def evaluate(self, parameters: NDArrays, config: Config) -> Tuple[float, int, dict[str, Scalar]]:
        """
        Evaluates the model on the validation set, and test set (if defined).

        Args:
            parameters (NDArrays): The parameters of the model to be evaluated.
            config (NDArrays): The config object from the server.

        Returns:
            Tuple[float, int, dict[str, Scalar]]: A loss associated with the evaluation, the number of samples in the
                validation/test set and the metric_values associated with evaluation.
        """
        if not self.initialized:
            self.setup_client(config)

        start_time = datetime.datetime.now()
        current_server_round = narrow_dict_type(config, "current_server_round", int)

        self.set_parameters(parameters, config, fitting_round=False)
        loss, metrics = self.validate()
        end_time = datetime.datetime.now()
        elapsed = end_time - start_time

        # Checkpoint based on the loss and metrics produced during validation AFTER server-side aggregation
        # NOTE: This assumes that the loss returned in the checkpointing loss
        self._maybe_checkpoint(loss, metrics, CheckpointMode.POST_AGGREGATION)

        self.reports_manager.report(
            {
                "eval_metrics": metrics,
                "eval_loss": loss,
                "eval_start": str(start_time),
                "eval_time_elapsed": str(elapsed),
                "eval_end": str(end_time),
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
            self.checkpointer is not None and self.checkpointer.pre_aggregation is not None
        )
        return evaluate_after_fit or pre_aggregation_checkpointing_enabled

    def _log_header_str(
        self,
        current_round: Optional[int] = None,
        current_epoch: Optional[int] = None,
        logging_mode: LoggingMode = LoggingMode.TRAIN,
    ) -> None:
        """
        Logs a header string. By default this is logged at the beginning of each local
        epoch or at the beginning of the round if training by steps

        Args:
            current_round (Optional[int], optional): The current FL round. (Ie current
                server round). Defaults to None.
            current_epoch (Optional[int], optional): The current epoch of local
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
        current_round: Optional[int] = None,
        current_epoch: Optional[int] = None,
        logging_mode: LoggingMode = LoggingMode.TRAIN,
    ) -> None:
        """
        Handles the logging of losses, metrics, and other information to the
        output file. Called only at the end of an epoch or server round

        Args:
            loss_dict (dict[str, float]): A dictionary of losses to log.
            metrics_dict (dict[str, Scalar]): A dictionary of the metric to log.
            current_round (Optional[int]): The current FL round (i.e., current server round).
            current_epoch (Optional[int]): The current epoch of local training.
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
        current_round: Optional[int],
        current_epoch: Optional[int],
        logging_mode: LoggingMode,
    ) -> Tuple[str, list[Tuple[LogLevel, str]]]:
        """
        This function can be overridden to provide any client specific
        information to the basic client logging. For example, perhaps a client
        uses an LR scheduler and wants the LR to be logged each epoch. Called at the
        beginning and end of each server round or local epoch. Also called at the end
        of validation/testing.

        Args:
            current_round (Optional[int]): The current FL round (i.e., current
                server round).
            current_epoch (Optional[int]): The current epoch of local training.
            logging_mode (LoggingMode): The logging mode (Training,
                Validation, or Testing).

        Returns:
            Optional[str]: A string to append to the header log string that
                typically announces the current server round and current epoch at the
                beginning of each round or local epoch.
            Optional[list[Tuple[LogLevel, str]]]]: A list of tuples where the
                first element is a LogLevel as defined in fl4health.utils.
                typing and the second element is a string message. Each item
                in the list will be logged at the end of each server round or epoch.
                Elements will also be logged at the end of validation/testing.
        """
        return "", []

    def get_client_specific_reports(self) -> dict[str, Any]:
        """
        This function can be overridden by an inheriting client to report
        additional client specific information to the wandb_reporter

        Returns:
            dict[str, Any]: A dictionary of things to report
        """
        return {}

    def _move_data_to_device(
        self, data: Union[TorchInputType, TorchTargetType]
    ) -> Union[TorchTargetType, TorchInputType]:
        """
        Moving data to self.device where data is intended to be either input to
        the model or the targets that the model is trying to achieve

        Args:
            data (TorchInputType | TorchTargetType): The data to move to
                self.device. Can be a TorchInputType or a TorchTargetType

        Raises:
            TypeError: Raised if data is not one of the types specified by
                TorchInputType or TorchTargetType

        Returns:
            Union[TorchTargetType, TorchInputType]: The data argument except now it's been moved to self.device
        """
        # Currently we expect both inputs and targets to be either tensors
        # or dictionaries of tensors
        if isinstance(data, torch.Tensor):
            return data.to(self.device)
        elif isinstance(data, dict):
            return {key: value.to(self.device) for key, value in data.items()}
        else:
            raise TypeError(
                "data must be of type torch.Tensor or dict[str, torch.Tensor]. \
                    If definition of TorchInputType or TorchTargetType has \
                    changed this method might need to be updated or split into \
                    two"
            )

    def is_empty_batch(self, input: Union[torch.Tensor, dict[str, torch.Tensor]]) -> bool:
        """
        Check whether input, which represents a batch of inputs to a model, is empty.

        Args:
            input (Union[torch.Tensor, dict[str, torch.Tensor]]): input batch.
            input can be of type torch.Tensor or dict[str, torch.Tensor], and in the
            latter case, the batch is considered to be empty if all tensors in the dictionary
            have length zero.

        Raises:
            TypeError: raised if input is not of type torch.Tensor or dict[str, torch.Tensor].
            ValueError: raised if input has type dict[str, torch.Tensor] and not all tensors
            within the dictionary have the same size.

        Returns:
            bool: True if input is an empty batch.
        """
        if isinstance(input, torch.Tensor):
            return len(input) == 0
        elif isinstance(input, dict):
            input_iter = iter(input.items())
            _, first_val = next(input_iter)
            first_val_len = len(first_val)
            if not all(len(val) == first_val_len for _, val in input_iter):
                raise ValueError("Not all tensors in the dictionary have the same size.")
            else:
                return first_val_len == 0
        else:
            raise TypeError("Input must be of type torch.Tensor or dict[str, torch.Tensor].")

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

    def train_step(self, input: TorchInputType, target: TorchTargetType) -> Tuple[TrainingLosses, TorchPredType]:
        """
        Given a single batch of input and target data, generate predictions, compute loss, update parameters and
        optionally update metrics if they exist. (ie backprop on a single batch of data).
        Assumes self.model is in train mode already.

        Args:
            input (TorchInputType): The input to be fed into the model.
            target (TorchTargetType): The target corresponding to the input.

        Returns:
            Tuple[TrainingLosses, TorchPredType]: The losses object from the train step along with
                a dictionary of any predictions produced by the model.
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

    def val_step(self, input: TorchInputType, target: TorchTargetType) -> Tuple[EvaluationLosses, TorchPredType]:
        """
        Given input and target, compute loss, update loss and metrics.
        Assumes self.model is in eval mode already.

        Args:
            input (TorchInputType): The input to be fed into the model.
            target (TorchTargetType): The target corresponding to the input.

        Returns:
            Tuple[EvaluationLosses, TorchPredType]: The losses object from the val step along with
            a dictionary of the predictions produced by the model.
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
        current_round: Optional[int] = None,
    ) -> Tuple[dict[str, float], dict[str, Scalar]]:
        """
        Train locally for the specified number of epochs.

        Args:
            epochs (int): The number of epochs for local training.
            current_round (Optional[int], optional): The current FL round.

        Returns:
            Tuple[dict[str, float], dict[str, Scalar]]: The loss and metrics dictionary from the local training.
                Loss is a dictionary of one or more losses that represent the different components of the loss.
        """
        self.model.train()
        steps_this_round = 0  # Reset number of steps this round
        report_data: dict[str, Any] = {"round": current_round}
        for local_epoch in range(epochs):
            self.train_metric_manager.clear()
            self.train_loss_meter.clear()
            # Print initial log string on epoch start
            self._log_header_str(current_round, local_epoch)
            # update before epoch hook
            self.update_before_epoch(epoch=local_epoch)
            # Update report data dict
            report_data.update({"fit_epoch": local_epoch})
            for input, target in self.maybe_progress_bar(self.train_loader):
                self.update_before_step(steps_this_round, current_round)
                # Assume first dimension is batch size. Sampling iterators (such as Poisson batch sampling), can
                # construct empty batches. We skip the iteration if this occurs.
                if self.is_empty_batch(input):
                    log(INFO, "Empty batch generated by data loader. Skipping step.")
                    continue

                input = self._move_data_to_device(input)
                target = self._move_data_to_device(target)
                losses, preds = self.train_step(input, target)
                self.train_loss_meter.update(losses)
                self.update_metric_manager(preds, target, self.train_metric_manager)
                self.update_after_step(steps_this_round, current_round)
                self.update_lr_schedulers(epoch=local_epoch)
                report_data.update({"fit_losses": losses.as_dict(), "fit_step": self.total_steps})
                report_data.update(self.get_client_specific_reports())
                self.reports_manager.report(report_data, current_round, local_epoch, self.total_steps)
                self.total_steps += 1
                steps_this_round += 1

            metrics = self.train_metric_manager.compute()
            loss_dict = self.train_loss_meter.compute().as_dict()

            # Log and report results
            self._log_results(loss_dict, metrics, current_round, local_epoch)
            report_data.update({"fit_metrics": metrics})
            report_data.update(self.get_client_specific_reports())
            self.reports_manager.report(report_data, current_round, local_epoch)

        # Return final training metrics
        return loss_dict, metrics

    def train_by_steps(
        self,
        steps: int,
        current_round: Optional[int] = None,
    ) -> Tuple[dict[str, float], dict[str, Scalar]]:
        """
        Train locally for the specified number of steps.

        Args:
            steps (int): The number of steps to train locally.
            current_round (Optional[int], optional): The current FL round

        Returns:
            Tuple[dict[str, float], dict[str, Scalar]]: The loss and metrics dictionary from the local training.
                Loss is a dictionary of one or more losses that represent the different components of the loss.
        """
        self.model.train()

        # Pass loader to iterator so we can step through train loader
        train_iterator = iter(self.train_loader)

        self.train_loss_meter.clear()
        self.train_metric_manager.clear()
        self._log_header_str(current_round)
        report_data: dict[str, Any] = {"round": current_round}
        for step in self.maybe_progress_bar(range(steps)):
            self.update_before_step(step, current_round)

            try:
                input, target = next(train_iterator)
            except StopIteration:
                # StopIteration is thrown if dataset ends
                # reinitialize data loader
                train_iterator = iter(self.train_loader)
                input, target = next(train_iterator)

            # Assume first dimension is batch size. Sampling iterators (such as Poisson batch sampling), can
            # construct empty batches. We skip the iteration if this occurs.
            if self.is_empty_batch(input):
                log(INFO, "Empty batch generated by data loader. Skipping step.")
                continue

            input = self._move_data_to_device(input)
            target = self._move_data_to_device(target)
            losses, preds = self.train_step(input, target)
            self.train_loss_meter.update(losses)
            self.update_metric_manager(preds, target, self.train_metric_manager)
            self.update_after_step(step, current_round)
            self.update_lr_schedulers(step=step)
            report_data.update({"fit_losses": losses.as_dict(), "fit_step": self.total_steps})
            report_data.update(self.get_client_specific_reports())
            self.reports_manager.report(report_data, current_round, None, self.total_steps)
            self.total_steps += 1

        loss_dict = self.train_loss_meter.compute().as_dict()
        metrics = self.train_metric_manager.compute()

        # Log and report results
        self._log_results(loss_dict, metrics, current_round)

        return loss_dict, metrics

    def _validate_or_test(
        self,
        loader: DataLoader,
        loss_meter: LossMeter,
        metric_manager: MetricManager,
        logging_mode: LoggingMode = LoggingMode.VALIDATION,
    ) -> Tuple[float, dict[str, Scalar]]:
        """
        Evaluate the model on the given validation or test dataset.

        Args:
            loader (DataLoader): The data loader for the dataset (validation or test).
            loss_meter (LossMeter): The meter to track the losses.
            metric_manager (MetricManager): The manager to track the metrics.
            logging_mode (LoggingMode): The LoggingMode for whether this evaluation is for validation or test.
              Default is for validation.

        Returns:
            Tuple[float, dict[str, Scalar]]: The loss and a dictionary of metrics from evaluation.
        """
        assert logging_mode in [
            LoggingMode.VALIDATION,
            LoggingMode.TEST,
        ], "logging_mode must be VALIDATION or TEST"
        self.model.eval()
        metric_manager.clear()
        loss_meter.clear()
        with torch.no_grad():
            for input, target in self.maybe_progress_bar(loader):
                input = self._move_data_to_device(input)
                target = self._move_data_to_device(target)
                losses, preds = self.val_step(input, target)
                loss_meter.update(losses)
                self.update_metric_manager(preds, target, metric_manager)

        # Compute losses and metrics over validation set
        loss_dict = loss_meter.compute().as_dict()
        metrics = metric_manager.compute()
        self._log_results(loss_dict, metrics, logging_mode=logging_mode)

        return loss_dict["checkpoint"], metrics

    def validate(self) -> Tuple[float, dict[str, Scalar]]:
        """
        Validate the current model on the entire validation
            and potentially an entire test dataset if it has been defined.

        Returns:
            Tuple[float, dict[str, Scalar]]: The validation loss and a dictionary of metrics
                from validation (and test if present).
        """
        val_loss, val_metrics = self._validate_or_test(self.val_loader, self.val_loss_meter, self.val_metric_manager)
        if self.test_loader:
            test_loss, test_metrics = self._validate_or_test(
                self.test_loader,
                self.test_loss_meter,
                self.test_metric_manager,
                LoggingMode.TEST,
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
            dict[str, Scalar]: A dictionary with two entries corresponding to the sample counts in
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

        # The following lines are type ignored because torch datasets are not "Sized"
        # IE __len__ is considered optionally defined. In practice, it is almost always defined
        # and as such, we will make that assumption.
        self.num_train_samples = len(self.train_loader.dataset)  # type: ignore
        self.num_val_samples = len(self.val_loader.dataset)  # type: ignore
        if self.test_loader:
            self.num_test_samples = len(self.test_loader.dataset)  # type: ignore

        self.set_optimizer(config)

        # Must initialize LR scheduler after parent method initializes optimizer
        # Add lr_scheduler to dictionary if user overrides get_lr_scheduler to return
        # scheduler for given optimizer
        self.lr_schedulers = {}
        for optimizer_key in self.optimizers.keys():
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
            ParameterExchanger: Used to exchange parameters between server and client.
        """
        return FullParameterExchanger()

    def predict(self, input: TorchInputType) -> Tuple[TorchPredType, TorchFeatureType]:
        """
        Computes the prediction(s), and potentially features, of the model(s) given the input.

        Args:
            input (TorchInputType): Inputs to be fed into the model. If input is
                of type dict[str, torch.Tensor], it is assumed that the keys of
                input match the names of the keyword arguments of self.model.
                forward().

        Returns:
            Tuple[TorchPredType, TorchFeatureType]: A tuple in which the
                first element contains a dictionary of predictions indexed by
                name and the second element contains intermediate activations
                indexed by name. By passing features, we can compute losses
                such as the contrastive loss in MOON. All predictions
                included in dictionary will by default be used to compute
                metrics separately.

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
        elif isinstance(output, torch.Tensor):
            return {"prediction": output}, {}
        elif isinstance(output, tuple):
            if len(output) != 2:
                raise ValueError(f"Output tuple should have length 2 but has length {len(output)}")
            preds, features = output
            return preds, features
        else:
            raise ValueError("Model forward did not return a tensor, dictionary of tensors, or tuple of tensors")

    def compute_loss_and_additional_losses(
        self, preds: TorchPredType, features: TorchFeatureType, target: TorchTargetType
    ) -> Tuple[torch.Tensor, Optional[dict[str, torch.Tensor]]]:
        """
        Computes the loss and any additional losses given predictions of the model and ground truth data.

        Args:
            preds (TorchPredType): Prediction(s) of the model(s) indexed by name.
            features (TorchFeatureType): Feature(s) of the model(s) indexed by name.
            target (TorchTargetType): Ground truth data to evaluate predictions against.

        Returns:
            Tuple[torch.Tensor, Union[dict[str, torch.Tensor], None]]; A tuple with:
                - The tensor for the loss
                - A dictionary of additional losses with their names and values, or None if
                    there are no additional losses.
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
            features: (TorchFeatureType): Feature(s) of the model(s) indexed by name.
            target: (TorchTargetType): Ground truth data to evaluate predictions against.

        Returns:
            TrainingLosses: an instance of TrainingLosses containing backward loss and additional losses
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
            features: (TorchFeatureType): Feature(s) of the model(s) indexed by name.
            target: (TorchTargetType): Ground truth data to evaluate predictions against.

        Returns:
            EvaluationLosses: an instance of EvaluationLosses containing checkpoint loss and additional losses
                indexed by name.
        """
        loss, additional_losses = self.compute_loss_and_additional_losses(preds, features, target)
        return EvaluationLosses(checkpoint=loss, additional_losses=additional_losses)

    def set_optimizer(self, config: Config) -> None:
        """
        Method called in the the setup_client method to set optimizer attribute returned by used-defined get_optimizer.
        In the simplest case, get_optimizer returns an optimizer. For more advanced use cases where a dictionary of
        string and optimizer are returned (ie APFL), the user must override this method.

        Args:
            config (Config): The config from the server.
        """
        optimizer = self.get_optimizer(config)
        assert not isinstance(optimizer, dict)
        self.optimizers = {"global": optimizer}

    def clone_and_freeze_model(self, model: nn.Module) -> nn.Module:
        """
        Creates a clone of the model with frozen weights to be used in loss calculations so the original model is
        preserved in its current state.

        Args:
            model (nn.Module): Model to clone and freeze
        Returns:
            nn.Module: Cloned and frozen model
        """

        cloned_model = copy.deepcopy(model)
        for param in cloned_model.parameters():
            param.requires_grad = False
        cloned_model.eval()

        return cloned_model

    def get_data_loaders(self, config: Config) -> Tuple[DataLoader, ...]:
        """
        User defined method that returns a PyTorch Train DataLoader
        and a PyTorch Validation DataLoader

        Args:
            config (Config): The config from the server.

        Returns:
            Tuple[DataLoader, ...]: Tuple of length 2. The client train and validation loader.

        Raises:
            NotImplementedError: To be defined in child class.
        """
        raise NotImplementedError

    def get_test_data_loader(self, config: Config) -> Optional[DataLoader]:
        """
        User defined method that returns a PyTorch Test DataLoader.
        By default, this function returns None, assuming that there is no test dataset to be used.
        If the user would like to load and evaluate a dataset,
            they need only override this function in their client class.

        Args:
            config (Config): The config from the server.

        Returns:
            Optional[DataLoader]. The optional client test loader. Returns None.

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
            TorchTargetType: Identical to target.
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

    def get_optimizer(self, config: Config) -> Union[Optimizer, dict[str, Optimizer]]:
        """
        Method to be defined by user that returns the PyTorch optimizer used to train models locally
        Return value can be a single torch optimizer or a dictionary of string and torch optimizer.
        Returning multiple optimizers is useful in methods like APFL which has a different optimizer
        for the local and global models.

        Args:
            config (Config): The config sent from the server.

        Returns:
            Union[Optimizer, dict[str, Optimizer]]: An optimizer or dictionary of optimizers to
            train model.

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
            nn.Module: The client model.

        Raises:
            NotImplementedError: To be defined in child class.
        """
        raise NotImplementedError

    def get_lr_scheduler(self, optimizer_key: str, config: Config) -> Union[None, _LRScheduler]:
        """
        Optional user defined method that returns learning rate scheduler
        to be used throughout training for the given optimizer. Defaults to None.

        Args:
            optimizer_key (str): The key in the optimizer dict corresponding
                to the optimizer we are optionally defining a learning rate
                scheduler for.
            config (Config): The config from the server.

        Returns:
            Union[None, _LRScheduler]: Client learning rate schedulers.
        """
        return None

    def update_lr_schedulers(self, step: Union[int, None] = None, epoch: Union[int, None] = None) -> None:
        """
        Updates any schedulers that exist. Can be overridden to customize update logic based on client state
            (ie self.total_steps).

        Args:
            step (Union[int, None]): If using local_steps, current step of this round. Otherwise None.
            epoch (Union[int, None]): If using local_epochs current epoch of this round. Otherwise None.
        """

        assert (step is None) ^ (epoch is None)

        for lr_scheduler in self.lr_schedulers.values():
            lr_scheduler.step()  # Update LR

    def update_before_train(self, current_server_round: int) -> None:
        """
        Hook method called before training with the number of current server rounds performed.
        NOTE: This method is called immediately AFTER the aggregated parameters are received from the server.
        For example, used by MOON and FENDA to save global modules after aggregation.

        Args:
            current_server_round (int): The number of current server round.
        """
        pass

    def update_after_train(self, local_steps: int, loss_dict: dict[str, float], config: Config) -> None:
        """
        Hook method called after training with the number of local_steps performed over the FL round and
        the corresponding loss dictionary. For example, used by Scaffold to update the control variates
        after a local round of training. Also used by FedProx to update the current loss based on the loss
        returned during training. Also used by MOON and FENDA to save trained modules weights before
        aggregation.

        Args:
            local_steps (int): The number of steps so far in the round in the local
                training.
            loss_dict (dict[str, float]): A dictionary of losses from local training.
            config (Config): The config from the server
        """
        pass

    def update_before_step(self, step: int, current_round: Optional[int] = None) -> None:
        """
        Hook method called before local train step.

        Args:
            step (int): The local training step that was most recently
                completed. Resets only at the end of the round.
            current_round (Optional[int], optional): The current FL server round
        """
        pass

    def update_after_step(self, step: int, current_round: Optional[int] = None) -> None:
        """
        Hook method called after local train step on client. step is an integer that represents
        the local training step that was most recently completed. For example, used by the APFL
        method to update the alpha value after a training a step. Also used by the MOON, FENDA
        and Ditto to update optimized beta value for MK-MMD loss after n steps.

        Args:
            step (int): The step number in local training that was most recently
                completed. Resets only at the end of the round.
            current_round (Optional[int], optional): The current FL server round
        """
        pass

    def update_before_epoch(self, epoch: int) -> None:
        """
        Hook method called before local epoch on client. Only called if client
        is being trained by epochs (ie. using local_epochs key instead of local
        steps in the server config file)

        Args:
            epoch (int): Integer representing the epoch about to begin
        """
        pass

    def maybe_progress_bar(self, iterable: Iterable) -> Iterable:
        """
        Used to print progress bars during client training and validation. If
        self.progress_bar is false, just returns the original input iterable
        without modifying it.

        Args:
            iterable (Iterable): The iterable to wrap

        Returns:
            Iterable: an iterator which acts exactly like the original
                iterable, but prints a dynamically updating progress bar every
                time a value is requested. Or the original iterable if
                self.progress_bar is False
        """
        if not self.progress_bar:
            return iterable
        else:
            # Create a clean looking tqdm instance that matches the flwr logging
            kwargs: Any = {
                "leave": True,
                "ascii": " >=",
                # "desc": f"{LOG_COLORS['INFO']}INFO{LOG_COLORS['RESET']} ",
                "unit": "steps",
                "dynamic_ncols": True,
                "bar_format": f"{LOG_COLORS['INFO']}INFO{LOG_COLORS['RESET']}" + " :        {l_bar}{bar}{r_bar}",
            }
            return tqdm(iterable, **kwargs)

    def transform_gradients(self, losses: TrainingLosses) -> None:
        """
        Hook function for model training only called after backwards pass but before
        optimizer step. Useful for transforming the gradients (such as with gradient
        clipping) before they are applied to the model weights.

        Args:
            losses (TrainingLosses): The losses object from the train step
        """
        pass

    def save_client_state(self) -> None:
        """
        Saves checkpoint dict consisting of client name, total steps, lr schedulers,
            metrics reporter and optimizers state. Method can be overridden to augment saved checkpointed state.
        """

        assert self.per_round_checkpointer is not None

        ckpt = {
            "lr_schedulers_state": {key: scheduler.state_dict() for key, scheduler in self.lr_schedulers.items()},
            "total_steps": self.total_steps,
            "client_name": self.client_name,
            "reports_manager": self.reports_manager,
            "optimizers_state": {key: optimizer.state_dict()["state"] for key, optimizer in self.optimizers.items()},
        }

        self.per_round_checkpointer.save_checkpoint(ckpt)

        log(
            INFO,
            f"Saving client state to checkpoint at {self.per_round_checkpointer.checkpoint_path}",
        )

    def load_client_state(self) -> None:
        """
        Load checkpoint dict consisting of client name, total steps, lr schedulers, metrics
            reporter and optimizers state. Method can be overridden to augment loaded checkpointed state.
        """
        assert self.per_round_checkpointer is not None and self.per_round_checkpointer.checkpoint_exists()

        ckpt = self.per_round_checkpointer.load_checkpoint()

        narrow_dict_type_and_set_attribute(self, ckpt, "client_name", "client_name", str)
        narrow_dict_type_and_set_attribute(self, ckpt, "total_steps", "total_steps", int)
        narrow_dict_type_and_set_attribute(self, ckpt, "reports_manager", "reports_manager", ReportsManager)

        assert "lr_schedulers_state" in ckpt and isinstance(ckpt["lr_schedulers_state"], dict)
        assert "optimizers_state" in ckpt and isinstance(ckpt["optimizers_state"], dict)

        # Optimizer is updated in setup_client to reference model weights from server
        # Thus, only optimizer state (per parameter values such as momentum)
        # should be loaded
        for key, optimizer in self.optimizers.items():
            optimizer_state = ckpt["optimizers_state"][key]
            optimizer_state_dict = optimizer.state_dict()
            optimizer_state_dict["state"] = optimizer_state
            optimizer.load_state_dict(optimizer_state_dict)

        # Schedulers initialized in setup_client to reference correct optimizers
        # Here we load in all other aspects of the scheduler state
        for key in self.lr_schedulers:
            self.lr_schedulers[key].load_state_dict(ckpt["lr_schedulers_state"][key])
