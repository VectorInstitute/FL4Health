import random
import string
from pathlib import Path
from typing import Optional, Type, TypeVar

import torch
import torch.nn as nn
from flwr.client import NumPyClient
from flwr.common import Config, NDArrays

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
        self.model_weights_initialized = False

        # Optional variable to store the weights that the client was initialized with during each round of training
        self.initial_weights: Optional[NDArrays] = None
        self.wandb_reporter: Optional[ClientWandBReporter] = None
        self.checkpointer: Optional[TorchCheckpointer] = None

    def generate_hash(self, length: int = 8) -> str:
        return "".join(random.choice(string.ascii_lowercase) for i in range(length))

    def _maybe_checkpoint(self, comparison_metric: float) -> None:
        if self.checkpointer:
            self.checkpointer.maybe_checkpoint(self.model, comparison_metric)

    def get_parameters(self, config: Config) -> NDArrays:
        """
        Determines which weights are sent back to the server for aggregation. This uses a parameter exchanger to
        determine parameters sent
        Args:
            config (Config): The config is sent by the FL server to allow for customization in the function if desired

        Returns:
            NDArrays: These are the parameters to be sent to the server. At minimum they represent the relevant model
                parameters to be aggregated, but can contain more information
        """

        assert self.model is not None and self.parameter_exchanger is not None
        return self.parameter_exchanger.push_parameters(self.model, config=config)

    def set_parameters(self, parameters: NDArrays, config: Config) -> None:
        """
        Sets the local model parameters transfered from the server using a parameter exchanger to coordinate how
        parameters are set. If it's the first time the model is being initialized, we assume the full model is being
        initialized and use the FullParameterExchanger() to set all model weights
        Args:
            parameters (NDArrays): Parameters have information about model state to be added to the relevant client
                model but may contain more information than that.
            config (Config): The config is sent by the FL server to allow for customization in the function if desired.
        """
        assert self.model is not None
        if not self.model_weights_initialized:
            self.initialize_all_model_weights(parameters, config)
        else:
            assert self.parameter_exchanger is not None
            self.parameter_exchanger.pull_parameters(parameters, self.model, config)

    def initialize_all_model_weights(self, parameters: NDArrays, config: Config) -> None:
        """
        If this is the first time we're initializing the model weights, we use the FullParameterExchanger to
        initialize all model components

        Args:
            parameters (NDArrays): Model parameters to be injected into the client model
            config (Config): The config is sent by the FL server to allow for customization in the function if desired.
        """
        FullParameterExchanger().pull_parameters(parameters, self.model, config)
        self.model_weights_initialized = True

    def narrow_config_type(self, config: Config, config_key: str, narrow_type_to: Type[T]) -> T:
        if config_key not in config:
            raise ValueError(f"{config_key} is not present in the Config.")

        config_value = config[config_key]
        if isinstance(config_value, narrow_type_to):
            return config_value
        else:
            raise ValueError(f"Provided configuration key ({config_key}) value does not have correct type")

    def shutdown(self) -> None:
        if self.wandb_reporter:
            self.wandb_reporter.shutdown_reporter()

    def setup_client(self, config: Config) -> None:
        """
        This method is used to set up all of the required components for the client through the config passed
        by the server and need only be done once. The Basic Client setup_client overrides this method to setup client
        by calling the user defined methods and setting the required attributes.
        """
        self.initialized = True
