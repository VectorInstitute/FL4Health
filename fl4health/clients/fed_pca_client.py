import random
import string
from pathlib import Path
from typing import Dict, Tuple, Type, TypeVar

from flwr.client.numpy_client import NumPyClient
from flwr.common import Config, NDArray, NDArrays, Scalar

from fl4health.parameter_exchange.parameter_packer import PrincipalComponentsPacker
from fl4health.PCA.pca import ClientSidePCA

T = TypeVar("T")


class FedPCAClient(NumPyClient):
    def __init__(self, data_path: Path, pc_path: Path, n_components: int) -> None:
        self.client_name = self.generate_hash()
        self.pca_packer = PrincipalComponentsPacker()
        self.initialized = False
        self.data_path = data_path
        self.pc_path = pc_path
        self._pca = ClientSidePCA(n_components=n_components)

    def generate_hash(self, length: int = 8) -> str:
        return "".join(random.choice(string.ascii_lowercase) for i in range(length))

    def get_parameters(self, config: Config) -> NDArrays:
        """
        _summary_

        Args:
            config (Config): The config is sent by the FL server to allow for customization in the function if desired

        Returns:
            NDArrays: packed principal components and their corresponding eigenvalues.
        """
        pcs, eigenvals = self._pca.get_pcs()
        return self.pca_packer.pack(pcs, eigenvals)

    def set_parameters(self, packed_pcs: NDArrays, config: Config) -> None:
        """
        Sets the merged principal components transfered from the server.

        Args:
            packed_pcs (NDArrays): principal components and their corresponding eigenvalues packed together.
            config (Config): The config is sent by the FL server to allow for customization in the function if desired.
        """
        pcs, eigenvals = self.pca_packer.unpack(packed_pcs)
        self._pca.update_pcs(pcs, eigenvals)
        self._pca.save_pcs(self.pc_path)

    def fit(self, parameters: NDArrays, config: Config) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """Compute PCA using the locally held dataset.

        Parameters
        ----------
        parameters : NDArrays
            This is unused in the case of PCA.
        config : Dict[str, Scalar]
            Configuration parameters which allow the
            server to influence training on the client. It can be used to
            communicate arbitrary values from the server to the client, for
            example, to set the number of (local) training epochs.

        Returns
        -------
        packed_pcs : NDArrays
            Principal components of local dataset along with their explained variance.
        num_examples : int
            The number of examples used for training.
        metrics : Dict[str, Scalar]
            A dictionary mapping arbitrary string keys to values of type
            bool, bytes, float, int, or str. It can be used to communicate
            arbitrary values back to the server.
        """
        X = self.get_data_numpy()
        principal_components, eigenvalues = self._pca.compute_pc(X)
        packed_pcs = self.pca_packer.pack(principal_components, eigenvalues)
        num_examples = X.shape[0]
        return packed_pcs, num_examples, {}

    def get_data_numpy(self) -> NDArray:
        """
        Return this client's data as a numpy array so PCA can be computed.

        Raises:
            NotImplementedError: This method should be implemented by the user
            based on the specific dataset encountered.

        Returns:
            X (NDArray): This client's data as a numpy array.
        """
        raise NotImplementedError

    def narrow_config_type(self, config: Config, config_key: str, narrow_type_to: Type[T]) -> T:
        if config_key not in config:
            raise ValueError(f"{config_key} is not present in the Config.")

        config_value = config[config_key]
        if isinstance(config_value, narrow_type_to):
            return config_value
        else:
            raise ValueError(f"Provided configuration key ({config_key}) value does not have correct type")

    def setup_client(self, config: Config) -> None:
        """
        This method is used to set up all of the required components for the client through the config passed
        by the server and need only be done once. The Basic Client setup_client overrides this method to setup client
        by calling the user defined methods and setting the required attributes.
        """
        self.initialized = True

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[float, int, Dict[str, Scalar]]:
        """Evaluate the provided parameters, in this case, principal components.

        Parameters
        ----------
        parameters : NDArrays
            The current (global) model parameters.
        config : Dict[str, Scalar]
            Configuration parameters which allow the server to influence
            evaluation on the client. It can be used to communicate
            arbitrary values from the server to the client, for example,
            to influence the number of examples used for evaluation.

        Returns
        -------
        loss : float
            The evaluation loss of the model on the local dataset.
        num_examples : int
            The number of examples used for evaluation.
        metrics : Dict[str, Scalar]
            A dictionary mapping arbitrary string keys to values of
            type bool, bytes, float, int, or str. It can be used to
            communicate arbitrary values back to the server.
        """
        # here we opted for returning the cumulative explained variance.
        self.set_parameters(packed_pcs=parameters, config=config)
        pcs, explained_variances = self.pca_packer.unpack(parameters)
        cumulative_explained_variance = sum(explained_variances)
        return 0.0, 0, {"cumulative_explained_variance": cumulative_explained_variance}
