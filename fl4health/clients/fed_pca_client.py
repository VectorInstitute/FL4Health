import random
import string
from logging import INFO, WARNING
from pathlib import Path

import torch
from flwr.client.numpy_client import NumPyClient
from flwr.common import Config, NDArrays, Scalar
from flwr.common.logger import log
from torch import Tensor
from torch.utils.data import DataLoader

from fl4health.model_bases.pca import PcaModule
from fl4health.parameter_exchange.full_exchanger import FullParameterExchanger
from fl4health.utils.config import narrow_dict_type


class FedPCAClient(NumPyClient):
    def __init__(
        self, data_path: Path, device: torch.device, model_save_dir: Path, client_name: str | None = None
    ) -> None:
        """
        Client that facilitates the execution of federated PCA.

        Args:
            data_path (Path): path to the data to be used to load the data for client-side training.
            device (torch.device): Device indicator for where to send the model, batches, labels etc. Often "cpu" or
                "cuda".
            model_save_dir (Path): Dir to save the PCA components for use later, perhaps in dimensionality reduction
            client_name (str | None, optional): client name, mainly used for saving components. Defaults to None.
        """
        self.client_name = self.generate_hash() if client_name is None else client_name
        self.model: PcaModule
        self.initialized = False
        self.data_path = data_path
        self.model_save_dir = model_save_dir
        self.device = device
        self.train_data_tensor: Tensor
        self.val_data_tensor: Tensor
        self.num_train_samples: int
        self.num_val_samples: int
        self.parameter_exchanger = FullParameterExchanger()

    def generate_hash(self, length: int = 8) -> str:
        """
        Generates unique hash used as id for client.

        Args:
            length (int, optional): Length of the generated hash. Defaults to 8.

        Returns:
            (str): Generated hash of length ``length``
        """
        return "".join(random.choice(string.ascii_lowercase) for _ in range(length))

    def get_parameters(self, config: Config) -> NDArrays:
        """
        Sends all of the model components back to the server. The model in this case represents the principal
        components that have been computed.

        Args:
            config (Config): Configurations to allow for customization of this functions behavior

        Returns:
            (NDArrays): Parameters representing the principal components computed by the client that need to be
                aggregated in some way.
        """
        if not self.initialized:
            log(INFO, "Setting up client and providing full model parameters to the server for initialization")
            if not config:
                log(
                    WARNING,
                    (
                        "This client has not yet been initialized and the config is empty. This may cause unexpected "
                        "failures, as setting up a client typically requires several configuration parameters, "
                        "including batch_size and current_server_round."
                    ),
                )

            # If initialized is False, the server is requesting model parameters from which to initialize all other
            # clients. As such get_parameters is being called before fit or evaluate, so we must call
            # setup_client first.
            self.setup_client(config)

            # Need all parameters even if normally exchanging partial
            return FullParameterExchanger().push_parameters(self.model, config=config)

        assert self.model is not None and self.parameter_exchanger is not None
        return self.parameter_exchanger.push_parameters(self.model, config=config)

    def set_parameters(self, parameters: NDArrays, config: Config) -> None:
        """
        Sets the merged principal components transferred from the server. Since federated PCA only runs for one round,
        the principal components obtained here are in fact the final result, so they are saved locally by each client
        for downstream tasks.

        Args:
            parameters (NDArrays): Aggregated principal components from the server. These are **FINAL** in the sense
                that FedPCA only runs for one round.
            config (Config): Configurations to allow for customization of this functions behavior
        """
        self.parameter_exchanger.pull_parameters(parameters, self.model, config)
        self.save_model()

    def get_data_loaders(self, config: Config) -> tuple[DataLoader, DataLoader]:
        """
        User defined method that returns a PyTorch Train ``DataLoader`` and a PyTorch Validation ``DataLoader``.

        Args:
            config (Config): Configurations sent by the server to allow for customization of this functions behavior.

        Returns:
            (tuple[DataLoader, DataLoader]): Tuple of length 2. The client train and validation loader.

        Raises:
            NotImplementedError: To be defined in child class.
        """
        raise NotImplementedError

    def get_model(self, config: Config) -> PcaModule:
        """
        Returns an instance of the ``PCAModule``. This module is used to facilitate FedPCA training on the client side.

        Args:
            config (Config): Configurations sent by the server to allow for customization of this functions behavior.

        Returns:
            (PcaModule): Module that determines how local FedPCA optimization will be performed.
        """
        low_rank = narrow_dict_type(config, "low_rank", bool)
        full_svd = narrow_dict_type(config, "full_svd", bool)
        rank_estimation = narrow_dict_type(config, "rank_estimation", int)
        return PcaModule(low_rank, full_svd, rank_estimation).to(self.device)

    def setup_client(self, config: Config) -> None:
        """
        Used to setup all of the components necessary to run ``FedPCA``.

        Args:
            config (Config): Configurations sent by the server to allow for customization of this functions behavior.
        """
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
        """
        This function should be used to "collate" each of the dataloader batches into a single monolithic tensor
        representing all of the data in the loader.

        Args:
            data_loader (DataLoader): The dataloader that can be used to iterate through a dataset

        Raises:
            NotImplementedError: Should be defined by the child class

        Returns:
            (Tensor): Single torch tensor representing all of the data stacked together.
        """
        raise NotImplementedError

    def fit(self, parameters: NDArrays, config: Config) -> tuple[NDArrays, int, dict[str, Scalar]]:
        """
        Function to perform the local side of ``FedPCA``. We don't use any parameters sent by the server. Hence
        ``parameters`` is ignored. We need only the ``train_data_tensor`` to do the work.

        Args:
            parameters (NDArrays): ignored
            config (Config): Configurations sent by the server to allow for customization of this functions behavior.

        Returns:
            (tuple[NDArrays, int, dict[str, Scalar]]): The local principal components following the local training
                along with the number of samples in the local training dataset and the computed metrics throughout the
                fit.
        """
        if not self.initialized:
            self.setup_client(config)
        center_data = narrow_dict_type(config, "center_data", bool)

        principal_components, singular_values = self.model(self.train_data_tensor, center_data)
        self.model.set_principal_components(principal_components, singular_values)

        cumulative_explained_variance = self.model.compute_cumulative_explained_variance()
        explained_variance_ratios = self.model.compute_explained_variance_ratios()
        metrics: dict[str, Scalar] = {
            "cumulative_explained_variance": cumulative_explained_variance,
            "top_explained_variance_ratio": explained_variance_ratios[0].item(),
        }
        return (self.get_parameters(config), self.num_train_samples, metrics)

    def evaluate(self, parameters: NDArrays, config: Config) -> tuple[float, int, dict[str, Scalar]]:
        """
        Evaluate merged principal components on the local validation set.

        Args:
            parameters (NDArrays): Server-merged principal components.
            config (Config): Configurations sent by the server to allow for customization of this functions behavior.

        Returns:
            (tuple[float, int, dict[str, Scalar]]): A loss associated with the evaluation, the number of samples in the
                validation/test set and the ``metric_values`` associated with evaluation.
        """
        if not self.initialized:
            self.setup_client(config)
            self.model.set_data_mean(self.model.maybe_reshape(self.train_data_tensor))
        self.set_parameters(parameters, config)
        num_components_eval = (
            narrow_dict_type(config, "num_components_eval", int) if "num_components_eval" in config else None
        )
        val_data_tensor_prepared = self.model.center_data(self.model.maybe_reshape(self.val_data_tensor)).to(
            self.device
        )
        reconstruction_loss = self.model.compute_reconstruction_error(val_data_tensor_prepared, num_components_eval)
        projection_variance = self.model.compute_projection_variance(val_data_tensor_prepared, num_components_eval)
        metrics: dict[str, Scalar] = {"projection_variance": projection_variance}
        return (reconstruction_loss, self.num_val_samples, metrics)

    def save_model(self) -> None:
        """
        Method to save the FedPCA computed principal components to disk. These can be reloaded to allow for
        dimensionality reduction in subsequent FL training.
        """
        final_model_save_path = self.model_save_dir / f"client_{self.client_name}_pca.pt"
        torch.save(self.model, final_model_save_path)
        log(INFO, f"Model parameters saved to {final_model_save_path}.")
