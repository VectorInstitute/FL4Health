import random
import string
from logging import INFO
from pathlib import Path
from typing import Dict, Tuple, Type, TypeVar

import torch
from flwr.client.numpy_client import NumPyClient
from flwr.common import Config, NDArrays, Scalar
from flwr.common.logger import log
from torch import Tensor
from torch.utils.data import DataLoader

from fl4health.model_bases.pca import PcaModule
from fl4health.parameter_exchange.full_exchanger import FullParameterExchanger

T = TypeVar("T")


class FedPCAClient(NumPyClient):
    def __init__(self, data_path: Path, device: torch.device, model_save_path: Path) -> None:
        """
        Client that facilitates the execution of federated PCA.
        """
        self.client_name = self.generate_hash()
        self.model: PcaModule
        self.initialized = False
        self.data_path = data_path
        self.model_save_path = model_save_path
        self.device = device
        self.train_data_tensor: Tensor
        self.val_data_tensor: Tensor
        self.num_train_samples: int
        self.num_val_samples: int
        self.parameter_exchanger = FullParameterExchanger()

    def generate_hash(self, length: int = 8) -> str:
        return "".join(random.choice(string.ascii_lowercase) for _ in range(length))

    def get_parameters(self, config: Config) -> NDArrays:
        assert self.model is not None and self.parameter_exchanger is not None
        return self.parameter_exchanger.push_parameters(self.model, config=config)

    def set_parameters(self, parameters: NDArrays, config: Config) -> None:
        """
        Sets the merged principal components transfered from the server.
        Since federated PCA only runs for one round, the principal components obtained here
        are in fact the final result, so they are saved locally by each client for downstream tasks.
        """
        self.parameter_exchanger.pull_parameters(parameters, self.model, config)
        self.save_model()

    def narrow_config_type(self, config: Config, config_key: str, narrow_type_to: Type[T]) -> T:
        if config_key not in config:
            raise ValueError(f"{config_key} is not present in the Config.")

        config_value = config[config_key]
        if isinstance(config_value, narrow_type_to):
            return config_value
        else:
            raise ValueError(f"Provided configuration key ({config_key}) value does not have correct type")

    def get_data_loaders(self, config: Config) -> Tuple[DataLoader, DataLoader]:
        """
        User defined method that returns a PyTorch Train DataLoader
        and a PyTorch Validation DataLoader
        """
        raise NotImplementedError

    def get_model(self, config: Config) -> PcaModule:
        """
        Returns an instance of the PCAModule.
        """
        low_rank = self.narrow_config_type(config, "low_rank", bool)
        full_svd = self.narrow_config_type(config, "full_svd", bool)
        rank_estimation = self.narrow_config_type(config, "rank_estimation", int)
        return PcaModule(low_rank, full_svd, rank_estimation).to(self.device)

    def setup_client(self, config: Config) -> None:
        self.model = self.get_model(config).to(self.device)

        train_loader, val_loader = self.get_data_loaders(config)
        self.train_data_tensor = self.get_data_tensor(train_loader).to(self.device)
        self.val_data_tensor = self.get_data_tensor(val_loader).to(self.device)

        # The following lines are type ignored because torch datasets are not "Sized"
        # IE __len__ is considered optionally defined. In practice, it is almost always defined
        # and as such, we will make that assumption.
        self.num_train_samples = len(train_loader.dataset)  # type: ignore
        self.num_val_samples = len(val_loader.dataset)  # type: ignore

        self.initialized = True

    def get_data_tensor(self, data_loader: DataLoader) -> Tensor:
        raise NotImplementedError

    def fit(self, parameters: NDArrays, config: Config) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """Perform PCA using the locally held dataset."""
        if not self.initialized:
            self.setup_client(config)
        center_data = self.narrow_config_type(config, "center_data", bool)

        principal_components, singular_values = self.model(self.train_data_tensor, center_data)
        self.model.set_principal_components(principal_components, singular_values)

        cumulative_explained_variance = self.model.compute_cumulative_explained_variance()
        explained_variance_ratios = self.model.compute_explained_variance_ratios()
        metrics: Dict[str, Scalar] = {
            "cumulative_explained_variance": cumulative_explained_variance,
            "top_explained_variance_ratio": explained_variance_ratios[0].item(),
        }
        return (self.get_parameters(config), self.num_train_samples, metrics)

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[float, int, Dict[str, Scalar]]:
        """
        Evaluate merged principal components on the local validation set.

        Args:
            parameters (NDArrays): Server-merged principal components.
            config (Dict[str, Scalar]): Config file.
        """
        if not self.initialized:
            self.setup_client(config)
            self.model.set_data_mean(self.model.maybe_reshape(self.train_data_tensor))
        self.set_parameters(parameters, config)
        num_components_eval = (
            self.narrow_config_type(config, "num_components_eval", int)
            if "num_components_eval" in config.keys()
            else None
        )
        val_data_tensor_prepared = self.model.center_data(self.model.maybe_reshape(self.val_data_tensor)).to(
            self.device
        )
        reconstruction_loss = self.model.compute_reconstruction_error(val_data_tensor_prepared, num_components_eval)
        projection_variance = self.model.compute_projection_variance(val_data_tensor_prepared, num_components_eval)
        metrics: Dict[str, Scalar] = {"projection_variance": projection_variance}
        return (reconstruction_loss, self.num_val_samples, metrics)

    def save_model(self) -> None:
        final_model_save_path = f"{self.model_save_path}/client_{self.generate_hash()}_pca.pt"
        torch.save(self.model, final_model_save_path)
        log(INFO, f"Model parameters saved to {final_model_save_path}.")
