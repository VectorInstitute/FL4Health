from abc import abstractmethod
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Type, TypeVar

import torch
import torch.nn as nn
from flwr.client import NumPyClient
from flwr.common.typing import Config, NDArrays, Scalar
from torch.utils.data import DataLoader

from fl4health.clients.basic_client import TorchInputType
from fl4health.reporting.metrics import MetricsReporter
from fl4health.utils.metrics import Metric, MetricManager

T = TypeVar("T")


class ModelMergeClient(NumPyClient):
    def __init__(
        self,
        data_path: Path,
        model_path: Path,
        metrics: Sequence[Metric],
        device: torch.device,
        metrics_reporter: Optional[MetricsReporter] = None,
    ) -> None:
        self.data_path = data_path
        self.model_path = model_path
        self.metrics = metrics
        self.device = device
        self.metrics_reporter = metrics_reporter

        self.initialized = False

        self.test_metric_manager = MetricManager(metrics=self.metrics, metric_manager_name="test")

        self.model: nn.Module
        self.test_loader: DataLoader

    @abstractmethod
    def get_model(self, config: Config) -> nn.Module:
        raise NotImplementedError

    @abstractmethod
    def get_test_dataloader(self, config: Config) -> DataLoader:
        raise NotImplementedError

    def setup_client(self, config: Config) -> None:
        self.model = self.get_model(config)
        self.test_loader = self.get_test_dataloader(config)

        self.initialized = True

    def get_parameters(self, config: Config) -> NDArrays:
        assert self.model is not None
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: NDArrays, config: Config) -> None:
        if not self.initialized:
            self.setup_client(config)

        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters: NDArrays, config: Config) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        self.setup_client(config)
        return self.get_parameters(config), 0, {}

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

    def validate(self) -> Dict[str, Scalar]:
        self.test_metric_manager.clear()
        with torch.no_grad():
            for input, target in self.test_loader:
                input, target = self._move_input_data_to_device(input).target.to(self.device)
                preds = self.model(target)
                self.test_metric_manager.update(preds, target)

        return self.test_metric_manager.compute()

    def evaluate(self, parameters: NDArrays, config: Config) -> Tuple[float, int, Dict[str, Scalar]]:
        self.set_parameters(parameters, config)
        metrics = self.validate()
        return 0, len(self.test_loader), metrics
