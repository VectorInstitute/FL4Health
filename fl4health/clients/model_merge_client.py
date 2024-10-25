import datetime
from abc import abstractmethod
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from flwr.client import NumPyClient
from flwr.common.typing import Config, NDArrays, Scalar
from torch.utils.data import DataLoader

from fl4health.parameter_exchange.full_exchanger import FullParameterExchanger
from fl4health.parameter_exchange.parameter_exchanger_base import ParameterExchanger
from fl4health.reporting.base_reporter import BaseReporter
from fl4health.reporting.reports_manager import ReportsManager
from fl4health.utils.metrics import Metric, MetricManager
from fl4health.utils.random import generate_hash
from fl4health.utils.typing import TorchInputType, TorchTargetType


class ModelMergeClient(NumPyClient):
    def __init__(
        self,
        data_path: Path,
        model_path: Path,
        metrics: Sequence[Metric],
        device: torch.device,
        reporters: Optional[Sequence[BaseReporter]] = None,
        client_name: Optional[str] = None,
    ) -> None:
        """
        ModelMergeClient to support functionality to simply perform model merging across client
            models and subsequently evaluate.

        Args:
            data_path (Path): path to the data to be used to load the data for client-side training
            model_path (Path): path to the checkpoint of the client model to be used in model merging.
            metrics (Sequence[Metric]): Metrics to be computed based on the labels and predictions of the client model
            device (torch.device): Device indicator for where to send the model, batches, labels etc. Often 'cpu' or
                'cuda'
            reporters (Sequence[BaseReporter], optional): A sequence of FL4Health
                reporters which the client should send data to.
            client_name (str): An optional client name that uniquely identifies a client.
                If not passed, a hash is randomly generated.
        """
        self.data_path = data_path
        self.model_path = model_path
        self.metrics = metrics
        self.device = device
        self.client_name = client_name if client_name is not None else generate_hash()

        self.initialized = False
        self.test_metric_manager = MetricManager(metrics=self.metrics, metric_manager_name="test")

        # Initialize reporters with client information.
        self.reports_manager = ReportsManager(reporters)
        self.reports_manager.initialize(id=self.client_name)

        self.model: nn.Module
        self.test_loader: DataLoader
        self.num_test_samples: int

    def setup_client(self, config: Config) -> None:
        """
        Sets up Merge Client by initializing model, dataloader and parameter exchanger
            with user defined methods. Subsquently, sets initialized attribute to True.

        Args:
            config (Config): The configuration from the server.
        """
        self.model = self.get_model(config)
        self.test_loader = self.get_test_data_loader(config)
        self.num_test_samples = len(self.test_loader.dataset)  # type: ignore
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
            only called once after model merging has occurred and before federated evaluation.

        Args:
            parameters (NDArrays): Parameters have information about model state to be added to the relevant client
                model but may contain more information than that.
            config (Config): The config is sent by the FL server to allow for customization in the function if desired.
        """
        assert self.initialized
        self.parameter_exchanger.pull_parameters(parameters, self.model)

    def fit(self, parameters: NDArrays, config: Config) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """
        Initializes client, validates local client model on local test data and returns parameters,
            test dataset length and test metrics. Importantly, parameters from Server, which is empty,
            is not used to initialized the client model.

        Note: Since we only assume the client provides a test_loader, client evaluation and sample
            counts are always based off the client test_loader.

        Args:
            parameters (NDArrays): Not used.
            config (NDArrays): The config from the server.

        Returns:
            Tuple[NDArrays, int, Dict[str, Scalar]]: The local model parameters along with the
                number of samples in the local test dataset and the computed metrics of the local model
                on the local test dataset.

        Raises:
            AssertionError: If model is initialized prior to fit method being called which
                should not happen in the case of the ModelMergeClient.
        """
        assert not self.initialized
        self.setup_client(config)

        self.reports_manager.report(
            data={"host_type": "client", "fit_start": datetime.datetime.now()},
        )

        val_metrics = self.validate()

        self.reports_manager.report(
            data={"fit_metrics": val_metrics, "host_type": "client", "fit_end": datetime.datetime.now()},
        )

        return self.get_parameters(config), self.num_test_samples, val_metrics

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
                "data must be of type torch.Tensor or Dict[str, torch.Tensor]. \
                    If definition of TorchInputType or TorchTargetType has \
                    changed this method might need to be updated or split into \
                    two"
            )

    def validate(self) -> Dict[str, Scalar]:
        """
        Validate the model on the test dataset.

        Returns:
            Tuple[float, Dict[str, Scalar]]: The loss and a dictionary of metrics
                from test set.
        """
        self.model.eval()
        self.test_metric_manager.clear()
        with torch.no_grad():
            for input, target in self.test_loader:
                input = self._move_data_to_device(input)
                target = self._move_data_to_device(target)
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
            DataLoader: Client test data loader.
        """
        raise NotImplementedError
