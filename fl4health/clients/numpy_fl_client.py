import random
import string
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Type, TypeVar

import torch
import torch.nn as nn
from flwr.client import NumPyClient
from flwr.common import Config, NDArrays, Scalar
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from fl4health.checkpointing.checkpointer import TorchCheckpointer
from fl4health.parameter_exchange.full_exchanger import FullParameterExchanger
from fl4health.parameter_exchange.parameter_exchanger_base import ParameterExchanger
from fl4health.reporting.fl_wanb import ClientWandBReporter

T = TypeVar("T")


class NumpyFlClient(NumPyClient):
    def __init__(self, data_path: Path, device: torch.device) -> None:
        self.client_name = self.generate_hash()
        self.model: nn.Module
        self.parameter_exchanger: ParameterExchanger
        self.initialized = False
        self.data_path = data_path
        self.device = device

        self.train_loader: DataLoader
        self.val_loader: DataLoader
        self.criterion: _Loss
        self.num_train_examples: int
        self.num_val_examples: int
        # Optional variable to store the weights that the client was initialized with during each round of training
        self.initial_weights: Optional[NDArrays] = None
        self.wandb_reporter: Optional[ClientWandBReporter] = None
        self.checkpointer: Optional[TorchCheckpointer] = None

    def generate_hash(self, length: int = 8) -> str:
        return "".join(random.choice(string.ascii_lowercase) for i in range(length))

    def _maybe_log_metrics(self, to_log: Dict[str, Any]) -> None:
        if self.wandb_reporter:
            self.wandb_reporter.report_metrics(to_log)

    def _maybe_checkpoint(self, comparison_metric: float) -> None:
        if self.checkpointer:
            self.checkpointer.maybe_checkpoint(self.model, comparison_metric)

    def setup_client(self, config: Config) -> None:
        """
        Set dataloaders, optimizers, parameter exchangers and other attributes derived from these.
        """
        self.model = self.get_model(config)
        train_loader, val_loader = self.get_data_loaders(config)
        self.train_loader = train_loader
        self.val_loader = val_loader

        num_train_samples = len(self.train_loader.dataset)  # type: ignore
        num_val_samples = len(self.val_loader.dataset)  # type: ignore
        self.num_train_samples = num_train_samples
        self.num_val_samples = num_val_samples

        self.optimizer = self.get_optimizer(self.model, config)
        self.parameter_exchanger = self.get_parameter_exchanger(config)
        self.criterion = self.get_criterion(config)

        self.initialized = True

    def get_data_loaders(self, config: Config) -> Tuple[DataLoader, DataLoader]:
        """
        User defined method that returns a PyTorch Train DataLoader
        and optionally a PyTorch Validation DataLoader
        """
        raise NotImplementedError

    def get_criterion(self, config: Config) -> _Loss:
        """
        Method that the user defines returning the criterion
        """
        raise NotImplementedError

    def get_optimizer(self, model: nn.Module, config: Config) -> Optimizer:
        """
        Method to be defined by user that returns the PyTorch optimizer used to train models locally
        """
        raise NotImplementedError

    def get_parameter_exchanger(self, config: Config) -> ParameterExchanger:
        """
        Returns Full Parameter Exchangers. Subclasses that require custom Parameter Exchangers can override this.
        """
        return FullParameterExchanger()

    def get_model(self, config: Config) -> nn.Module:
        """
        User defined method that Returns PyTorch model
        """
        raise NotImplementedError

    def get_parameters(self, config: Config) -> NDArrays:
        # Determines which weights are sent back to the server for aggregation. This uses a parameter exchanger to
        # determine parameters sent
        assert self.model is not None and self.parameter_exchanger is not None
        return self.parameter_exchanger.push_parameters(self.model, config=config)

    def set_parameters(self, parameters: NDArrays, config: Config) -> None:
        # Sets the local model parameters transfered from the server using a parameter exchanger to coordinate how
        # parameters are set
        assert self.model is not None and self.parameter_exchanger is not None
        self.parameter_exchanger.pull_parameters(parameters, self.model, config)

    def narrow_config_type(self, config: Config, config_key: str, narrow_type_to: Type[T]) -> T:
        config_value = config[config_key]
        if isinstance(config_value, narrow_type_to):
            return config_value
        else:
            raise ValueError(f"Provided configuration key ({config_key}) value does not have correct type")

    def shutdown(self) -> None:
        if self.wandb_reporter:
            self.wandb_reporter.shutdown_reporter()

    def get_properties(self, config: Config) -> Dict[str, Scalar]:
        """
        Return properties (train and validation dataset sample counts) of client.
        """
        return {"num_train_samples": self.num_train_samples, "num_val_samples": self.num_val_samples}
