from pathlib import Path
from typing import Type, TypeVar

import torch
import torch.nn as nn
from flwr.client import NumPyClient
from flwr.common import Config, NDArrays

from fl4health.parameter_exchange.parameter_exchanger_base import ParameterExchanger

T = TypeVar("T")


class NumpyFlClient(NumPyClient):
    def __init__(self, data_path: Path, device: torch.device) -> None:
        self.model: nn.Module
        self.parameter_exchanger: ParameterExchanger
        self.initialized = False
        self.data_path = data_path
        self.device = device

    def setup_client(self, config: Config) -> None:
        """
        This method should be used to set up all of the required components for the client through the config passed
        by the server and need only be done once. The quintessential example is data loaders with a batch size set by
        the server in the config. The parameter initialized should be set to true when this function is finished.
        Overriding this class and calling super is the preferred flow.
        """
        self.initialized = True

    def get_parameters(self, config: Config) -> NDArrays:
        # Determines which weights are sent back to the server for aggregation. This uses a parameter exchanger to
        # determine parameters sent
        assert self.model is not None and self.parameter_exchanger is not None
        return self.parameter_exchanger.push_parameters(self.model, config)

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
