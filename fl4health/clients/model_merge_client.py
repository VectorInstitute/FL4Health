import datetime
import random
import string
from abc import abstractmethod
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Type, TypeVar

import torch
import torch.nn as nn
from flwr.client import NumPyClient
from flwr.common.typing import Config, NDArrays, Scalar
from torch.utils.data import DataLoader

from fl4health.clients.basic_client import TorchInputType
from fl4health.parameter_exchange.full_exchanger import FullParameterExchanger
from fl4health.parameter_exchange.parameter_exchanger_base import ParameterExchanger
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
        self.client_name = self.generate_hash()
        self.test_metric_manager = MetricManager(metrics=self.metrics, metric_manager_name="test")

        if metrics_reporter is not None:
            self.metrics_reporter = metrics_reporter
        else:
            self.metrics_reporter = MetricsReporter(run_id=self.client_name)

        self.model: nn.Module
        self.test_loader: DataLoader

    def generate_hash(self, length: int = 8) -> str:
        """
        Generates unique hash used as id for client.

        Args:
           length (int): The length of the hash.

        Returns:
            str: client id
        """
        return "".join(random.choice(string.ascii_lowercase) for _ in range(length))

    def setup_client(self, config: Config) -> None:
        """
        Sets up Merge Client by initializing model, dataloader and parameter exchanger
            with user defined methods. Subsquently, sets initialized attribute to True.

        Args:
            config (Config): The configuration from the server.
        """
        self.model = self.get_model(config)
        self.test_loader = self.get_test_data_loader(config)
        self.parameter_exchanger = self.get_parameter_exchanger(config)

        self.initialized = True

    def get_parameters(self, config: Config) -> NDArrays:
        """
        Determines which parameters are sent back to the server for aggregation.
            This uses a parameter exchanger to determine parameters sent.

        For the ModelMergeClient, we assume that self.setup_client has already been called
            as it does not support client polling so get_parameters is called from fit and
            thus should be initialized by this point.

        Args:
            config (Config): The config is sent by the FL server to allow for customization in the function if desired.

        Returns:
            NDArrays: These are the parameters to be sent to the server. At minimum they represent the relevant model
                parameters to be aggregated, but can contain more information.
        """
        assert self.model is not None
        return self.parameter_exchanger.push_parameters(self.model, config=config)

    def set_parameters(self, parameters: NDArrays, config: Config) -> None:
        """
        Sets the local model parameters transferred from the server using a parameter exchanger
            to coordinate how parameters are set.

        For the ModelMergeClient, we assume that initially parameters are being set to the parameters
            in the nn.Module returned by the user defined get_model method. Thus, set_parameters is
            only called once after model merging has occured and before federated evaluation.

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
        assert self.initialized
        self.parameter_exchanger.pull_parameters(parameters, self.model)

    def fit(self, parameters: NDArrays, config: Config) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """
        Initializes client, validates local client model on local test data and returns parameters,
            test dataset length and test metrics. Importantly, parameters from Server, which is empty,
            is not used to initialized the client model.

        Args:
            parameters (NDArrays): The parameters of the model to be used in fit.
            config (NDArrays): The config from the server.

        Returns:
            Tuple[NDArrays, int, Dict[str, Scalar]]: The parameters following the local training along with the
            number of samples in the local training dataset and the computed metrics throughout the fit.

        Raises:
            ValueError: If local_steps or local_epochs is not specified in config.
        """
        assert not self.initialized
        self.setup_client(config)
        assert self.metrics_reporter is not None
        self.metrics_reporter.add_to_metrics_at_round(
            1,
            data={"fit_start": datetime.datetime.now()},
        )

        val_metrics = self.validate()

        self.metrics_reporter.add_to_metrics_at_round(
            1,
            data={
                "fit_metrics": val_metrics,
            },
        )

        return self.get_parameters(config), len(self.test_loader.dataset), val_metrics  # type: ignore

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
        """
        Validate the current model on the entire validation
            and potentially an entire test dataset if it has been defined.

        Returns:
            Tuple[float, Dict[str, Scalar]]: The validation loss and a dictionary of metrics
                from validation (and test if present).
        """
        self.test_metric_manager.clear()
        with torch.no_grad():
            for input, target in self.test_loader:
                input, target = self._move_input_data_to_device(input), target.to(self.device)
                preds = {"predictions": self.model(input)}
                self.test_metric_manager.update(preds, target)

        return self.test_metric_manager.compute()

    def evaluate(self, parameters: NDArrays, config: Config) -> Tuple[float, int, Dict[str, Scalar]]:
        """
        Evaluate the provided parameters using the locally held dataset.

        Args:
            parameters (NDArrays): The current model parameters.
            config (Config): Configuration object from the server.

        Returns:
            Tuple[float, int, Dict[str, Scalar]: The float represents the
                loss which is assumed to be 0 for the ModelMergeClient.
                The int represents the number of examples in the local test dataset and the
                dictionary is the computed metrics on the test set.
        """
        self.set_parameters(parameters, config)
        metrics = self.validate()
        return 0.0, len(self.test_loader), metrics

    def get_parameter_exchanger(self, config: Config) -> ParameterExchanger:
        """
        Parameter exchange is assumed to always be full for model merging clients. However, this functionality
        may be overridden if a different exchanger is needed.

        Used in non-standard way for ModelMergClient as set_parameters is only called for evaluate as
            parameters should initially be set to the parameters in the nn.Module returned by get_model.

        Args:
            config (Config): Configuration object from the server.

        Returns:
            FullParameterExchanger: The parameter exchanger used to set and get parameters.
        """
        return FullParameterExchanger()

    @abstractmethod
    def get_model(self, config: Config) -> nn.Module:
        """
        User defined method that returns PyTorch model.
        This is the local model that will be communicated
        to the server for merging.

        Args:
            config (Config): The config from the server.

        Returns:
            nn.Module: The client model.

        Raises:
            NotImplementedError: To be defined in child class.
        """
        raise NotImplementedError

    @abstractmethod
    def get_test_data_loader(self, config: Config) -> DataLoader:
        """
        User defined method that returns a PyTorch Test DataLoader.

        Args:
            config (Config): The config from the server.

        Returns:
            DataLoader. Client test data loader.
        """
        raise NotImplementedError
