import random
import string
from pathlib import Path
from typing import Dict, Tuple, Type, TypeVar

import torch
from flwr.client.numpy_client import NumPyClient
from flwr.common import Config, NDArrays, Scalar
from torch import Tensor
from torch.utils.data import DataLoader

from fl4health.model_bases.pca import PCAModule
from fl4health.parameter_exchange.packing_exchanger import ParameterExchangerWithPacking
from fl4health.parameter_exchange.parameter_packer import ParameterPackerWithIntVar

T = TypeVar("T")


class FedPCAClient(NumPyClient):
    def __init__(self, data_path: Path, device: torch.device, model_save_path: Path) -> None:
        self.client_name = self.generate_hash()
        self.model: PCAModule
        self.initialized = False
        self.data_path = data_path
        self.model_save_path = model_save_path
        self.device = device
        self.train_loader: DataLoader
        self.val_loader: DataLoader
        self.num_train_samples: int
        self.num_val_samples: int
        self.parameter_exchanger: ParameterExchangerWithPacking[int]

    def generate_hash(self, length: int = 8) -> str:
        return "".join(random.choice(string.ascii_lowercase) for i in range(length))

    def get_parameters(self, config: Config) -> NDArrays:
        assert self.model is not None and self.parameter_exchanger is not None
        model_parameters = self.parameter_exchanger.push_parameters(self.model, config=config)
        return self.parameter_exchanger.pack_parameters(model_parameters, self.model.num_components)

    def set_parameters(self, parameters: NDArrays, config: Config) -> None:
        model_parameters, num_components = self.parameter_exchanger.unpack_parameters(parameters)
        self.model.num_components = num_components
        self.parameter_exchanger.pull_parameters(model_parameters, self.model, config)

    def narrow_config_type(self, config: Config, config_key: str, narrow_type_to: Type[T]) -> T:
        if config_key not in config:
            raise ValueError(f"{config_key} is not present in the Config.")

        config_value = config[config_key]
        if isinstance(config_value, narrow_type_to):
            return config_value
        else:
            raise ValueError(f"Provided configuration key ({config_key}) value does not have correct type")

    def get_parameter_exchanger(self, config: Config) -> ParameterExchangerWithPacking[int]:
        return ParameterExchangerWithPacking(ParameterPackerWithIntVar())

    def get_data_loaders(self, config: Config) -> Tuple[DataLoader, DataLoader]:
        """
        User defined method that returns a PyTorch Train DataLoader
        and a PyTorch Validation DataLoader
        """
        raise NotImplementedError

    def get_model(self, config: Config) -> PCAModule:
        """
        User defined method that returns an uninitialized instance of PCAModule.
        """
        raise NotImplementedError

    def setup_client(self, config: Config) -> None:
        self.model = self.get_model(config).to(self.device)

        train_loader, val_loader = self.get_data_loaders(config)
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.parameter_exchanger = self.get_parameter_exchanger(config)

        # The following lines are type ignored because torch datasets are not "Sized"
        # IE __len__ is considered optionally defined. In practice, it is almost always defined
        # and as such, we will make that assumption.
        self.num_train_samples = len(self.train_loader.dataset)  # type: ignore
        self.num_val_samples = len(self.val_loader.dataset)  # type: ignore

        self.initialized = True

    def get_data_tensor(self, data_loader: DataLoader) -> Tensor:
        raise NotImplementedError

    def evaluate_pca_train(self) -> Dict[str, Scalar]:
        """
        User defined method that evaluates the locally computed principal components on the training set.

        Returns:
            metrics (Dict[str, Scalar]): evaluation results.
        """
        raise NotImplementedError

    def evaluate_pca_val(self) -> Dict[str, Scalar]:
        """
        User defined method that evaluates the merged principal components on the validation set.

        Returns:
            metrics (Dict[str, Scalar]): evaluation results.
        """
        raise NotImplementedError

    def fit(self, parameters: NDArrays, config: Config) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """Perform PCA using the locally held dataset."""
        if not self.initialized:
            self.setup_client(config)
        train_data_tensor = self.get_data_tensor(self.train_loader)
        principal_components, eigenvalues = self.model(train_data_tensor)
        self.model.set_principal_components(principal_components, eigenvalues)
        metrics = self.evaluate_pca_train()

        return (self.get_parameters(config), self.num_train_samples, metrics)

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[float, int, Dict[str, Scalar]]:
        self.set_parameters(parameters, config)
        val_data_tensor = self.get_data_tensor(self.val_loader)
        reconstruction_loss = self.model.compute_reconstruction_loss(val_data_tensor)
        metrics = self.evaluate_pca_val()

        return (reconstruction_loss, self.num_val_samples, metrics)

    def save_model(self) -> None:
        torch.save(self.model.state_dict(), self.model_save_path)
