from logging import INFO
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from flwr.common.logger import log
from flwr.common.typing import Config, NDArrays, Scalar
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from fl4health.checkpointing.checkpointer import TorchCheckpointer
from fl4health.clients.numpy_fl_client import NumpyFlClient
from fl4health.parameter_exchange.full_exchanger import FullParameterExchanger
from fl4health.parameter_exchange.parameter_exchanger_base import ParameterExchanger
from fl4health.reporting.fl_wanb import ClientWandBReporter
from fl4health.utils.losses import Losses, LossMeter, LossMeterType
from fl4health.utils.metrics import Metric, MetricMeter, MetricMeterManager, MetricMeterType


class BasicClient(NumpyFlClient):
    """
    Base FL Client with functionality to train, evaluate, log, report and checkpoint.
    User is responsible for implementing methods: get_model, get_optimizer, get_data_loaders, get_criterion
    Other methods can be overriden to achieve custom functionality.
    """

    def __init__(
        self,
        data_path: Path,
        metrics: Sequence[Metric],
        device: torch.device,
        loss_meter_type: LossMeterType = LossMeterType.AVERAGE,
        metric_meter_type: MetricMeterType = MetricMeterType.AVERAGE,
        checkpointer: Optional[TorchCheckpointer] = None,
    ) -> None:
        super().__init__(data_path, device)
        self.metrics = metrics
        self.checkpointer = checkpointer
        self.train_loss_meter = LossMeter.get_meter_by_type(loss_meter_type)
        self.val_loss_meter = LossMeter.get_meter_by_type(loss_meter_type)

        # Define mapping from prediction key to meter to pass to MetricMeterManager constructor for train and val
        train_key_to_meter_map = {
            "prediction": MetricMeter.get_meter_by_type(self.metrics, metric_meter_type, "train_meter")
        }
        self.train_metric_meter_mngr = MetricMeterManager(train_key_to_meter_map)
        val_key_to_meter_map = {
            "prediction": MetricMeter.get_meter_by_type(self.metrics, metric_meter_type, "val_meter")
        }
        self.val_metric_meter_mngr = MetricMeterManager(val_key_to_meter_map)

        self.model: nn.Module
        self.optimizer: torch.optim.Optimizer

        self.train_loader: DataLoader
        self.val_loader: DataLoader
        self.num_train_samples: int
        self.num_val_samples: int
        self.learning_rate: float

        # Need to track total_steps across rounds for WANDB reporting
        self.total_steps: int = 0

    def process_config(self, config: Config) -> Tuple[Union[int, None], Union[int, None], int]:
        """
        Method to ensure the required keys are present in config and extracts the values.
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

        # Either local epochs or local steps is none based on what key is passed in the config
        return local_epochs, local_steps, current_server_round

    def fit(self, parameters: NDArrays, config: Config) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        local_epochs, local_steps, current_server_round = self.process_config(config)

        if not self.initialized:
            self.setup_client(config)

        self.set_parameters(parameters, config)

        if local_epochs is not None:
            loss_dict, metrics = self.train_by_epochs(local_epochs, current_server_round)
            local_steps = len(self.train_loader) * local_epochs  # total steps over training round
        elif local_steps is not None:
            loss_dict, metrics = self.train_by_steps(local_steps, current_server_round)
        else:
            raise ValueError("Must specify either local_epochs or local_steps in the Config.")

        # Update after train round (Used by Scaffold and DP-Scaffold Client to update control variates)
        self.update_after_train(local_steps, loss_dict)

        # FitRes should contain local parameters, number of examples on client, and a dictionary holding metrics
        # calculation results.
        return (
            self.get_parameters(config),
            self.num_train_samples,
            metrics,
        )

    def evaluate(self, parameters: NDArrays, config: Config) -> Tuple[float, int, Dict[str, Scalar]]:
        if not self.initialized:
            self.setup_client(config)

        self.set_parameters(parameters, config)
        loss, metric_values = self.validate()
        # EvaluateRes should return the loss, number of examples on client, and a dictionary holding metrics
        # calculation results.
        return (
            loss,
            self.num_val_samples,
            metric_values,
        )

    def _handle_logging(
        self,
        loss_dict: Dict[str, float],
        metrics_dict: Dict[str, Scalar],
        current_round: Optional[int] = None,
        current_epoch: Optional[int] = None,
        is_validation: bool = False,
    ) -> None:
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

        # If reporter is None we do not report to wandb and return
        if self.wandb_reporter is None:
            return

        # If no current_round is passed or current_round is None, set current_round to 0
        # This situation only arises when we do local finetuning and call train_by_epochs or train_by_steps explicitly
        current_round = current_round if current_round is not None else 0

        reporting_dict: Dict[str, Any] = {"server_round": current_round}
        reporting_dict.update({"step": self.total_steps})
        reporting_dict.update(loss_dict)
        reporting_dict.update(metric_dict)
        self.wandb_reporter.report_metrics(reporting_dict)

    def train_step(self, input: torch.Tensor, target: torch.Tensor) -> Tuple[Losses, Dict[str, torch.Tensor]]:
        """
        Given input and target, generate predictions, compute loss, optionally update metrics if they exist.
        Assumes self.model is in train model already.
        """
        # Clear gradients from optimizer if they exist
        self.optimizer.zero_grad()

        # Call user defined methods to get predictions and compute loss
        preds = self.predict(input)
        losses = self.compute_loss(preds, target)

        # Compute backward pass and update paramters with optimizer
        losses.backward.backward()
        self.optimizer.step()

        return losses, preds

    def val_step(self, input: torch.Tensor, target: torch.Tensor) -> Tuple[Losses, Dict[str, torch.Tensor]]:
        """
        Given input and target, compute loss, update loss and metrics
        Assumes self.model is in eval mode already.
        """

        # Get preds and compute loss
        with torch.no_grad():
            preds = self.predict(input)
            losses = self.compute_loss(preds, target)

        return losses, preds

    def train_by_epochs(
        self, epochs: int, current_round: Optional[int] = None
    ) -> Tuple[Dict[str, float], Dict[str, Scalar]]:
        self.model.train()
        local_step = 0
        for local_epoch in range(epochs):
            self.train_metric_meter_mngr.clear()
            self.train_loss_meter.clear()
            for input, target in self.train_loader:
                input, target = input.to(self.device), target.to(self.device)
                losses, preds = self.train_step(input, target)
                self.train_loss_meter.update(losses)
                self.train_metric_meter_mngr.update(preds, target)
                self.update_after_step(local_step)
                self.total_steps += 1
                local_step += 1
            metrics = self.train_metric_meter_mngr.compute()
            losses = self.train_loss_meter.compute()
            loss_dict = losses.as_dict()

            # Log results and maybe report via WANDB
            self._handle_logging(loss_dict, metrics, current_round=current_round, current_epoch=local_epoch)
            self._handle_reporting(loss_dict, metrics, current_round=current_round)

        # Return final training metrics
        return loss_dict, metrics

    def train_by_steps(
        self, steps: int, current_round: Optional[int] = None
    ) -> Tuple[Dict[str, float], Dict[str, Scalar]]:
        self.model.train()

        # Pass loader to iterator so we can step through train loader
        train_iterator = iter(self.train_loader)

        self.train_loss_meter.clear()
        self.train_metric_meter_mngr.clear()
        for step in range(steps):
            try:
                input, target = next(train_iterator)
            except StopIteration:
                # StopIteration is thrown if dataset ends
                # reinitialize data loader
                train_iterator = iter(self.train_loader)
                input, target = next(train_iterator)

            input, target = input.to(self.device), target.to(self.device)
            losses, preds = self.train_step(input, target)
            self.train_loss_meter.update(losses)
            self.train_metric_meter_mngr.update(preds, target)
            self.update_after_step(step)
            self.total_steps += 1

        losses = self.train_loss_meter.compute()
        loss_dict = losses.as_dict()
        metrics = self.train_metric_meter_mngr.compute()

        # Log results and maybe report via WANDB
        self._handle_logging(loss_dict, metrics, current_round=current_round)
        self._handle_reporting(loss_dict, metrics, current_round=current_round)

        return loss_dict, metrics

    def validate(self) -> Tuple[float, Dict[str, Scalar]]:
        self.model.eval()
        self.val_metric_meter_mngr.clear()
        self.val_loss_meter.clear()
        with torch.no_grad():
            for input, target in self.val_loader:
                input, target = input.to(self.device), target.to(self.device)
                losses, preds = self.val_step(input, target)
                self.val_loss_meter.update(losses)
                self.val_metric_meter_mngr.update(preds, target)

        # Compute losses and metrics over validation set
        losses = self.val_loss_meter.compute()
        loss_dict = losses.as_dict()
        metrics = self.val_metric_meter_mngr.compute()
        self._handle_logging(loss_dict, metrics, is_validation=True)

        # Checkpoint based on loss which is output of user defined compute_loss method
        self._maybe_checkpoint(loss_dict["checkpoint"])
        return loss_dict["checkpoint"], metrics

    def get_properties(self, config: Config) -> Dict[str, Scalar]:
        """
        Return properties (train and validation dataset sample counts) of client.
        """
        if not self.initialized:
            self.setup_client(config)

        return {"num_train_samples": self.num_train_samples, "num_val_samples": self.num_val_samples}

    def setup_client(self, config: Config) -> None:
        """
        Set dataloaders, optimizers, parameter exchangers and other attributes derived from these.
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

        optimizer = self.get_optimizer(config)
        assert isinstance(optimizer, Optimizer)
        self.optimizer = optimizer

        self.learning_rate = self.optimizer.defaults["lr"]
        self.criterion = self.get_criterion(config)
        self.parameter_exchanger = self.get_parameter_exchanger(config)

        self.wandb_reporter = ClientWandBReporter.from_config(self.client_name, config)

        super().setup_client(config)

    def get_parameter_exchanger(self, config: Config) -> ParameterExchanger:
        """
        Returns Full Parameter Exchangers. Subclasses that require custom Parameter Exchangers can override this.
        """
        return FullParameterExchanger()

    def predict(self, input: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Return dict of str and torch.Tensor contaiing predictions when given input.
        In the default case, the dict has a single item with key prediction.
        In more complicated approaches such as APFL, the dict has as many items as prediction types
        User can override for more complex logic.
        """
        preds = self.model(input)
        preds = preds if isinstance(preds, dict) else {"prediction": preds}
        return preds

    def compute_loss(self, preds: Dict[str, torch.Tensor], target: torch.Tensor) -> Losses:
        """
        Computes loss given preds and torch and the user defined criterion. Optionally includes dictionairy of
        loss components if you wish to train the total loss as well as sub losses if they exist.
        Predicitons are a dictionairy of str and torch.Tensor. In the base case we have one set of prediction
        stored in the prediction key of the dict.
        For more complicated loss computations (additional loss components or multiple prediction types)
        this method should be overridden.
        """
        loss = self.criterion(preds["prediction"], target)
        losses = Losses(checkpoint=loss, backward=loss)
        return losses

    def get_data_loaders(self, config: Config) -> Tuple[DataLoader, DataLoader]:
        """
        User defined method that returns a PyTorch Train DataLoader
        and a PyTorch Validation DataLoader
        """
        raise NotImplementedError

    def get_criterion(self, config: Config) -> _Loss:
        """
        User defined method that returns PyTorch loss to train model.
        """
        raise NotImplementedError

    def get_optimizer(self, config: Config) -> Union[Optimizer, Dict[str, Optimizer]]:
        """
        Method to be defined by user that returns the PyTorch optimizer used to train models locally
        Return value can be a single torch optimizer or a dictionary of string and torch optimizer.
        """
        raise NotImplementedError

    def get_model(self, config: Config) -> nn.Module:
        """
        User defined method that Returns PyTorch model
        """
        raise NotImplementedError

    def update_after_train(self, local_steps: int, loss_dict: Dict[str, float]) -> None:
        """
        Called after training with the number of local_steps performed over the FL round and
        the corresponding loss dictionairy.
        """
        pass

    def update_after_step(self, step: int) -> None:
        """
        Called after local train step on client. step is an integer that represents
        the local training step that was most recently completed.
        """
        pass
