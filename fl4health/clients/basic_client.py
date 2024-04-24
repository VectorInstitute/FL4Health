import copy
import datetime
import random
import string
from logging import INFO
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Type, TypeVar, Union

import torch
import torch.nn as nn
from flwr.client import NumPyClient
from flwr.common.logger import log
from flwr.common.typing import Config, NDArrays, Scalar
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from fl4health.checkpointing.client_module import CheckpointMode, ClientCheckpointModule
from fl4health.parameter_exchange.full_exchanger import FullParameterExchanger
from fl4health.parameter_exchange.parameter_exchanger_base import ParameterExchanger
from fl4health.reporting.fl_wandb import ClientWandBReporter
from fl4health.reporting.metrics import MetricsReporter
from fl4health.utils.losses import EvaluationLosses, LossMeter, LossMeterType, TrainingLosses
from fl4health.utils.metrics import Metric, MetricManager

T = TypeVar("T")
TorchInputType = TypeVar("TorchInputType", torch.Tensor, Dict[str, torch.Tensor])


class BasicClient(NumPyClient):
    def __init__(
        self,
        data_path: Path,
        metrics: Sequence[Metric],
        device: torch.device,
        loss_meter_type: LossMeterType = LossMeterType.AVERAGE,
        checkpointer: Optional[ClientCheckpointModule] = None,
        metrics_reporter: Optional[MetricsReporter] = None,
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
            checkpointer (Optional[TorchCheckpointer], optional): Checkpointer to be used for client-side
                checkpointing. Defaults to None.
            metrics_reporter (Optional[MetricsReporter], optional): A metrics reporter instance to record the metrics
                during the execution. Defaults to an instance of MetricsReporter with default init parameters.
        """

        self.data_path = data_path
        self.device = device
        self.metrics = metrics
        self.checkpointer = checkpointer

        self.client_name = self.generate_hash()

        if metrics_reporter is not None:
            self.metrics_reporter = metrics_reporter
        else:
            self.metrics_reporter = MetricsReporter(run_id=self.client_name)

        self.initialized = False  # Whether or not the client has been setup

        # Loss and Metric management
        self.train_loss_meter = LossMeter[TrainingLosses](loss_meter_type, TrainingLosses)
        self.val_loss_meter = LossMeter[EvaluationLosses](loss_meter_type, EvaluationLosses)
        self.train_metric_manager = MetricManager(metrics=self.metrics, metric_manager_name="train")
        self.val_metric_manager = MetricManager(metrics=self.metrics, metric_manager_name="val")

        # Optional variable to store the weights that the client was initialized with during each round of training
        self.initial_weights: Optional[NDArrays] = None

        self.wandb_reporter: Optional[ClientWandBReporter] = None
        self.total_steps: int = 0  # Need to track total_steps across rounds for WANDB reporting

        # Attributes to be initialized in setup_client
        self.parameter_exchanger: ParameterExchanger
        self.model: nn.Module
        self.optimizers: Dict[str, torch.optim.Optimizer]
        self.train_loader: DataLoader
        self.val_loader: DataLoader
        self.num_train_samples: int
        self.num_val_samples: int
        self.learning_rate: Optional[float] = None

    def _maybe_checkpoint(self, loss: float, metrics: Dict[str, Scalar], checkpoint_mode: CheckpointMode) -> None:
        """
        If checkpointer exists, maybe checkpoint model based on the provided metric values.

        Args:
            loss (float): validation loss to potentially be used for checkpointing
            metrics (Dict[str, float]): validation metrics to potentially be used for checkpointing
        """
        if self.checkpointer:
            self.checkpointer.maybe_checkpoint(self.model, loss, metrics, checkpoint_mode)

    def generate_hash(self, length: int = 8) -> str:
        """
        Generates unique hash used as id for client.

        Args:
           length (int): The length of the hash.

        Returns:
            str: client id
        """
        return "".join(random.choice(string.ascii_lowercase) for i in range(length))

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
        current_server_round = self.narrow_config_type(config, "current_server_round", int)
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

    def narrow_config_type(self, config: Config, config_key: str, narrow_type_to: Type[T]) -> T:
        """
        Checks if a config_key exists in config and if so, verify it is of type narrow_type_to.

        Args:
            config (Config): The config object from the server.
            config_key (str): The key to check config for.
            narrow_type_to (Type[T]): The expected type of config[config_key]

        Returns:
            T: The type-checked value at config[config_key]

        Raises:
            ValueError: If config[config_key] is not of type narrow_type_to or
                if the config_key is not present in config.
        """
        if config_key not in config:
            raise ValueError(f"{config_key} is not present in the Config.")

        config_value = config[config_key]
        if isinstance(config_value, narrow_type_to):
            return config_value
        else:
            raise ValueError(f"Provided configuration key ({config_key}) value does not have correct type")

    def shutdown(self) -> None:
        """
        Shuts down the client. Involves shutting down W&B reporter if one exists.
        """
        if self.wandb_reporter:
            self.wandb_reporter.shutdown_reporter()

        self.metrics_reporter.add_to_metrics({"shutdown": datetime.datetime.now()})

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
        current_server_round = self.narrow_config_type(config, "current_server_round", int)

        if ("local_epochs" in config) and ("local_steps" in config):
            raise ValueError("Config cannot contain both local_epochs and local_steps. Please specify only one.")
        elif "local_epochs" in config:
            local_epochs = self.narrow_config_type(config, "local_epochs", int)
            local_steps = None
        elif "local_steps" in config:
            local_steps = self.narrow_config_type(config, "local_steps", int)
            local_epochs = None
        else:
            raise ValueError("Must specify either local_epochs or local_steps in the Config.")

        try:
            evaluate_after_fit = self.narrow_config_type(config, "evaluate_after_fit", bool)
        except ValueError:
            evaluate_after_fit = False

        # Either local epochs or local steps is none based on what key is passed in the config
        return local_epochs, local_steps, current_server_round, evaluate_after_fit

    def fit(self, parameters: NDArrays, config: Config) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """
        Processes config, initializes client (if first round) and performs training based on the passed config.

        Args:
            parameters (NDArrays): The parameters of the model to be used in fit.
            config (NDArrays): The config from the server.

        Returns:
            Tuple[NDArrays, int, Dict[str, Scalar]]: The parameters following the local training along with the
            number of samples in the local training dataset and the computed metrics throughout the fit.

        Raises:
            ValueError: If local_steps or local_epochs is not specified in config.
        """
        local_epochs, local_steps, current_server_round, evaluate_after_fit = self.process_config(config)

        if not self.initialized:
            self.setup_client(config)

        self.metrics_reporter.add_to_metrics_at_round(
            current_server_round,
            data={"fit_start": datetime.datetime.now()},
        )

        self.set_parameters(parameters, config, fitting_round=True)

        self.update_before_train(current_server_round)

        if local_epochs is not None:
            loss_dict, metrics = self.train_by_epochs(local_epochs, current_server_round)
            local_steps = len(self.train_loader) * local_epochs  # total steps over training round
        elif local_steps is not None:
            loss_dict, metrics = self.train_by_steps(local_steps, current_server_round)
        else:
            raise ValueError("Must specify either local_epochs or local_steps in the Config.")

        # Update after train round (Used by Scaffold and DP-Scaffold Client to update control variates)
        self.update_after_train(local_steps, loss_dict)

        # Check if we should run an evaluation with validation data after fit
        # (for example, this is used by FedDGGA)
        if self._should_evaluate_after_fit(evaluate_after_fit):
            validation_loss, validation_metrics = self.evaluate_after_fit()
            metrics.update(validation_metrics)
            # We perform a pre-aggregation checkpoint if applicable
            self._maybe_checkpoint(validation_loss, validation_metrics, CheckpointMode.PRE_AGGREGATION)

        self.metrics_reporter.add_to_metrics_at_round(
            current_server_round,
            data={
                "fit_metrics": metrics,
                "loss_dict": loss_dict,
            },
        )

        # FitRes should contain local parameters, number of examples on client, and a dictionary holding metrics
        # calculation results.
        return (
            self.get_parameters(config),
            self.num_train_samples,
            metrics,
        )

    def evaluate_after_fit(self) -> Tuple[float, Dict[str, Scalar]]:
        """
        Run self.validate right after fit to collect metrics on the local model against validation data.

        Returns: (Dict[str, Scalar]) a dictionary with the metrics.

        """
        loss, metric_values = self.validate()
        # The computed loss value is packed into the metrics dictionary, perhaps for use on the server-side
        metrics_after_fit = {
            **metric_values,  # type: ignore
            "val - loss": loss,
        }
        return loss, metrics_after_fit

    def evaluate(self, parameters: NDArrays, config: Config) -> Tuple[float, int, Dict[str, Scalar]]:
        """
        Evaluates the model on the validation set.

        Args:
            parameters (NDArrays): The parameters of the model to be evaluated.
            config (NDArrays): The config object from the server.

        Returns:
            Tuple[float, int, Dict[str, Scalar]]: A loss associated with the evaluation, the number of samples in the
                validation set and the metric_values associated with evaluation.
        """
        if not self.initialized:
            self.setup_client(config)

        current_server_round = self.narrow_config_type(config, "current_server_round", int)
        self.metrics_reporter.add_to_metrics_at_round(
            current_server_round,
            data={"evaluate_start": datetime.datetime.now()},
        )

        self.set_parameters(parameters, config, fitting_round=False)
        loss, metrics = self.validate()

        # Checkpoint based on the loss and metrics produced during validation AFTER server-side aggregation
        # NOTE: This assumes that the loss returned in the checkpointing loss
        self._maybe_checkpoint(loss, metrics, CheckpointMode.POST_AGGREGATION)

        self.metrics_reporter.add_to_metrics_at_round(
            current_server_round,
            data={
                "evaluate_metrics": metrics,
                "loss": loss,
            },
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

    def _handle_logging(
        self,
        loss_dict: Dict[str, float],
        metrics_dict: Dict[str, Scalar],
        current_round: Optional[int] = None,
        current_epoch: Optional[int] = None,
        is_validation: bool = False,
    ) -> None:
        """
        Handles the logging of losses, metrics and other information to the output file.

        Args:
            loss_dict (Dict[str, float]): A dictionary of losses to log.
            metrics_dict (Dict[str, Scalar]): A dictionary of the metric to log.
            current_round (Optional[int]): The current FL round (ie current server round).
            current_epoch (Optional[int]): The current epoch of local training.
            is_validation (bool): Whether or not this logging is for validation set.
        """
        initial_log_str = f"Current FL Round: {str(current_round)}\t" if current_round is not None else ""
        initial_log_str += f"Current Epoch: {str(current_epoch)}" if current_epoch is not None else ""
        if initial_log_str != "":
            log(INFO, initial_log_str)

        metric_string = "\t".join([f"{key}: {str(val)}" for key, val in metrics_dict.items()])
        loss_string = "\t".join([f"{key}: {str(val)}" for key, val in loss_dict.items()])
        metric_prefix = "Validation" if is_validation else "Training"
        log(
            INFO,
            f"Client {metric_prefix} Losses: {loss_string} \n" f"Client {metric_prefix} Metrics: {metric_string}",
        )

    def _handle_reporting(
        self,
        loss_dict: Dict[str, float],
        metric_dict: Dict[str, Scalar],
        current_round: Optional[int] = None,
    ) -> None:
        """
        Handles reporting of losses and metrics to W&B.
        Args:
            loss_dict (Dict[str, float]): A dictionary of losses to log.
            metrics_dict (Dict[str, Scalar]): A dictionary of metrics to log.
            current_round (Optional[int]): The current FL round.
        """
        # If reporter is None we do not report to wandb and return
        if self.wandb_reporter is None:
            return

        # If no current_round is passed or current_round is None, set current_round to 0
        # This situation only arises when we do local fine-tuning and call train_by_epochs or train_by_steps explicitly
        current_round = current_round if current_round is not None else 0

        reporting_dict: Dict[str, Any] = {"server_round": current_round}
        reporting_dict.update({"step": self.total_steps})
        reporting_dict.update(loss_dict)
        reporting_dict.update(metric_dict)
        self.wandb_reporter.report_metrics(reporting_dict)

    def _move_input_data_to_device(self, data: TorchInputType) -> TorchInputType:
        """
        Moving data to self.device, where data is intended to be the input to
        self.model's forward method.

        Args:
            data (TorchInputType): input data to the forward method of self.model.
            data can be of type torch.Tensor or Dict[str, torch.Tensor], and in the
            latter case, all tensors in the dictionary are moved to self.device.

        Raises:
            TypeError: raised if data is not of type torch.Tensor or Dict[str, torch.Tensor]
        """
        if isinstance(data, torch.Tensor):
            return data.to(self.device)
        elif isinstance(data, dict):
            return {key: value.to(self.device) for key, value in data.items()}
        else:
            raise TypeError("data must be of type torch.Tensor or Dict[str, torch.Tensor].")

    def is_empty_batch(self, input: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> bool:
        """
        Check whether input, which represents a batch of inputs to a model, is empty.

        Args:
            input (Union[torch.Tensor, Dict[str, torch.Tensor]]): input batch.
            input can be of type torch.Tensor or Dict[str, torch.Tensor], and in the
            latter case, the batch is considered to be empty if all tensors in the dictionary
            have length zero.

        Raises:
            TypeError: raised if input is not of type torch.Tensor or Dict[str, torch.Tensor].
            ValueError: raised if input has type Dict[str, torch.Tensor] and not all tensors
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
            raise TypeError("Input must be of type torch.Tensor or Dict[str, torch.Tensor].")

    def train_step(
        self, input: TorchInputType, target: torch.Tensor
    ) -> Tuple[TrainingLosses, Dict[str, torch.Tensor]]:
        """
        Given a single batch of input and target data, generate predictions, compute loss, update parameters and
        optionally update metrics if they exist. (ie backprop on a single batch of data).
        Assumes self.model is in train mode already.

        Args:
            input (TorchInputType): The input to be fed into the model.
            target (torch.Tensor): The target corresponding to the input.

        Returns:
            Tuple[TrainingLosses, Dict[str, torch.Tensor]]: The losses object from the train step along with
                a dictionary of any predictions produced by the model.
        """
        # Clear gradients from optimizer if they exist
        self.optimizers["global"].zero_grad()

        # Call user defined methods to get predictions and compute loss
        preds, features = self.predict(input)
        losses = self.compute_training_loss(preds, features, target)

        # Compute backward pass and update parameters with optimizer
        losses.backward["backward"].backward()
        self.optimizers["global"].step()

        return losses, preds

    def val_step(
        self, input: TorchInputType, target: torch.Tensor
    ) -> Tuple[EvaluationLosses, Dict[str, torch.Tensor]]:
        """
        Given input and target, compute loss, update loss and metrics.
        Assumes self.model is in eval mode already.

        Args:
            input (TorchInputType): The input to be fed into the model.
            target (torch.Tensor): The target corresponding to the input.

        Returns:
            Tuple[EvaluationLosses, Dict[str, torch.Tensor]]: The losses object from the val step along with
            a dictionary of the predictions produced by the model.
        """

        # Get preds and compute loss
        with torch.no_grad():
            preds, features = self.predict(input)
            losses = self.compute_evaluation_loss(preds, features, target)

        return losses, preds

    def train_by_epochs(
        self, epochs: int, current_round: Optional[int] = None
    ) -> Tuple[Dict[str, float], Dict[str, Scalar]]:
        """
        Train locally for the specified number of epochs.

        Args:
            epochs (int): The number of epochs for local training.
            current_round (Optional[int]): The current FL round.

        Returns:
            Tuple[Dict[str, float], Dict[str, Scalar]]: The loss and metrics dictionary from the local training.
                Loss is a dictionary of one or more losses that represent the different components of the loss.
        """
        self.model.train()
        local_step = 0
        for local_epoch in range(epochs):
            self.train_metric_manager.clear()
            self.train_loss_meter.clear()
            for input, target in self.train_loader:
                # Assume first dimension is batch size. Sampling iterators (such as Poisson batch sampling), can
                # construct empty batches. We skip the iteration if this occurs.
                if self.is_empty_batch(input):
                    log(INFO, "Empty batch generated by data loader. Skipping step.")
                    continue

                input, target = self._move_input_data_to_device(input), target.to(self.device)
                losses, preds = self.train_step(input, target)
                self.train_loss_meter.update(losses)
                self.train_metric_manager.update(preds, target)
                self.update_after_step(local_step)
                self.total_steps += 1
                local_step += 1
            metrics = self.train_metric_manager.compute()
            loss_dict = self.train_loss_meter.compute().as_dict()

            # Log results and maybe report via WANDB
            self._handle_logging(loss_dict, metrics, current_round=current_round, current_epoch=local_epoch)
            self._handle_reporting(loss_dict, metrics, current_round=current_round)

        # Return final training metrics
        return loss_dict, metrics

    def train_by_steps(
        self, steps: int, current_round: Optional[int] = None
    ) -> Tuple[Dict[str, float], Dict[str, Scalar]]:
        """
        Train locally for the specified number of steps.

        Args:
            steps (int): The number of steps to train locally.

        Returns:
            Tuple[Dict[str, float], Dict[str, Scalar]]: The loss and metrics dictionary from the local training.
                Loss is a dictionary of one or more losses that represent the different components of the loss.
        """
        self.model.train()

        # Pass loader to iterator so we can step through train loader
        train_iterator = iter(self.train_loader)

        self.train_loss_meter.clear()
        self.train_metric_manager.clear()
        for step in range(steps):
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

            input, target = self._move_input_data_to_device(input), target.to(self.device)
            losses, preds = self.train_step(input, target)
            self.train_loss_meter.update(losses)
            self.train_metric_manager.update(preds, target)
            self.update_after_step(step)
            self.total_steps += 1

        loss_dict = self.train_loss_meter.compute().as_dict()
        metrics = self.train_metric_manager.compute()

        # Log results and maybe report via WANDB
        self._handle_logging(loss_dict, metrics, current_round=current_round)
        self._handle_reporting(loss_dict, metrics, current_round=current_round)

        return loss_dict, metrics

    def validate(self) -> Tuple[float, Dict[str, Scalar]]:
        """
        Validate the current model on the entire validation dataset.

        Returns:
            Tuple[float, Dict[str, Scalar]]: The validation loss and a dictionary of metrics from validation.
        """
        self.model.eval()
        self.val_metric_manager.clear()
        self.val_loss_meter.clear()
        with torch.no_grad():
            for input, target in self.val_loader:
                input, target = self._move_input_data_to_device(input), target.to(self.device)
                losses, preds = self.val_step(input, target)
                self.val_loss_meter.update(losses)
                self.val_metric_manager.update(preds, target)

        # Compute losses and metrics over validation set
        loss_dict = self.val_loss_meter.compute().as_dict()
        metrics = self.val_metric_manager.compute()
        self._handle_logging(loss_dict, metrics, is_validation=True)

        return loss_dict["checkpoint"], metrics

    def get_properties(self, config: Config) -> Dict[str, Scalar]:
        """
        Return properties (train and validation dataset sample counts) of client.

        Args:
            config (Config): The config from the server.

        Returns:
            Dict[str, Scalar]: A dictionary with two entries corresponding to the sample counts in
                the train and validation set.
        """
        if not self.initialized:
            self.setup_client(config)

        return {"num_train_samples": self.num_train_samples, "num_val_samples": self.num_val_samples}

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

        # The following lines are type ignored because torch datasets are not "Sized"
        # IE __len__ is considered optionally defined. In practice, it is almost always defined
        # and as such, we will make that assumption.
        self.num_train_samples = len(self.train_loader.dataset)  # type: ignore
        self.num_val_samples = len(self.val_loader.dataset)  # type: ignore

        self.set_optimizer(config)

        self.criterion = self.get_criterion(config).to(self.device)
        self.parameter_exchanger = self.get_parameter_exchanger(config)

        self.wandb_reporter = ClientWandBReporter.from_config(self.client_name, config)

        self.metrics_reporter.add_to_metrics({"type": "client", "initialized": datetime.datetime.now()})

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

    def predict(self, input: TorchInputType) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Computes the prediction(s), and potentially features, of the model(s) given the input.

        Args:
            input (TorchInputType): Inputs to be fed into the model.
            If input is of type Dict[str, torch.Tensor], it is assumed that the keys of input
            match the names of the keyword arguments of self.model.forward().

        Returns:
            Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]: A tuple in which the first element
            contains predictions indexed by name and the second element contains intermediate activations
            index by name. By passing features, we can compute losses such as the model contrasting loss in MOON.
            All predictions included in dictionary will be used to compute metrics.

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
            raise TypeError('"input" must be of type torch.Tensor or Dict[str, torch.Tensor].')

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
        self,
        preds: Dict[str, torch.Tensor],
        features: Dict[str, torch.Tensor],
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, Union[Dict[str, torch.Tensor], None]]:
        """
        Computes the loss and any additional losses given predictions of the model and ground truth data.

        Args:
            preds (Dict[str, torch.Tensor]): Prediction(s) of the model(s) indexed by name.
            features (Dict[str, torch.Tensor]): Feature(s) of the model(s) indexed by name.
            target (torch.Tensor): Ground truth data to evaluate predictions against.

        Returns:
            Tuple[torch.Tensor, Union[Dict[str, torch.Tensor], None]]; A tuple with:
                - The tensor for the loss
                - A dictionary of additional losses with their names and values, or None if
                    there are no additional losses.
        """
        return self.criterion(preds["prediction"], target), None

    def compute_training_loss(
        self,
        preds: Dict[str, torch.Tensor],
        features: Dict[str, torch.Tensor],
        target: torch.Tensor,
    ) -> TrainingLosses:
        """
        Computes training loss given predictions (and potentially features) of the model and ground truth data.

        Args:
            preds (Dict[str, torch.Tensor]): Prediction(s) of the model(s) indexed by name. Anything stored
                in preds will be used to compute metrics.
            features: (Dict[str, torch.Tensor]): Feature(s) of the model(s) indexed by name.
            target: (torch.Tensor): Ground truth data to evaluate predictions against.

        Returns:
            TrainingLosses: an instance of TrainingLosses containing backward loss and additional losses
                indexed by name.
        """
        loss, additional_losses = self.compute_loss_and_additional_losses(preds, features, target)
        return TrainingLosses(backward=loss, additional_losses=additional_losses)

    def compute_evaluation_loss(
        self,
        preds: Dict[str, torch.Tensor],
        features: Dict[str, torch.Tensor],
        target: torch.Tensor,
    ) -> EvaluationLosses:
        """
        Computes evaluation loss given predictions (and potentially features) of the model and ground truth data.

        Args:
            preds (Dict[str, torch.Tensor]): Prediction(s) of the model(s) indexed by name. Anything stored
                in preds will be used to compute metrics.
            features: (Dict[str, torch.Tensor]): Feature(s) of the model(s) indexed by name.
            target: (torch.Tensor): Ground truth data to evaluate predictions against.

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
        """Clone and freeze model for use in various loss calculation.

        Args:
            model (nn.Module): model to clone and freeze

        Returns:
            nn.Module: cloned and frozen model
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

    def get_criterion(self, config: Config) -> _Loss:
        """
        User defined method that returns PyTorch loss to train model.

        Args:
            config (Config): The config from the server.

        Raises:
            NotImplementedError: To be defined in child class.
        """
        raise NotImplementedError

    def get_optimizer(self, config: Config) -> Union[Optimizer, Dict[str, Optimizer]]:
        """
        Method to be defined by user that returns the PyTorch optimizer used to train models locally
        Return value can be a single torch optimizer or a dictionary of string and torch optimizer.
        Returning multiple optimizers is useful in methods like APFL which has a different optimizer
        for the local and global models.

        Args:
            config (Config): The config sent from the server.

        Returns:
            Union[Optimizer, Dict[str, Optimizer]]: An optimizer or dictionary of optimizers to
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

    def update_before_train(self, current_server_round: int) -> None:
        """
        Hook method called before training with the number of current server rounds performed.
        For example, used by Moon and FENDA to save global modules after aggregation.

        Args:
            current_server_round (int): The number of current server round.
        """
        pass

    def update_after_train(self, local_steps: int, loss_dict: Dict[str, float]) -> None:
        """
        Hook method called after training with the number of local_steps performed over the FL round and
        the corresponding loss dictionary. For example, used by Scaffold to update the control variates
        after a local round of training. Also used by FedProx to update the current loss based on the loss
        returned during training. Also used by Moon and FENDA to save trained modules weights before
        aggregation.

        Args:
            local_steps (int): The number of steps in the local training.
            loss_dict (Dict[str, float]): A dictionary of losses from local training.
        """
        pass

    def update_after_step(self, step: int) -> None:
        """
        Hook method called after local train step on client. step is an integer that represents
        the local training step that was most recently completed. For example, used by the APFL
        method to update the alpha value after a training a step.

        Args:
            step (int): The step number in local training that was most recently completed.
        """
        pass
