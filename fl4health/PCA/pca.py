from logging import INFO
from pathlib import Path
from typing import Tuple

import numpy as np
from flwr.common.logger import log
from flwr.common.typing import NDArray, NDArrays
from sklearn.decomposition import PCA


class ClientPCA:
    def __init__(self, n_components: int) -> None:
        self._pca = PCA(n_components=n_components, random_state=42)
        self.principal_components: NDArray
        self.eigenvalues: NDArray

    def compute_pc(self, X: NDArray) -> Tuple[NDArray, NDArray]:
        self._pca.fit(X)
        components, eigen_vals = self._pca.components_, self._pca.explained_variance_
        return components, eigen_vals

    def pc_projection(self, X: NDArray) -> NDArray:
        return self.principal_components @ X

    def save_principal_components(self, principal_components: NDArray, eigenvalues: NDArray, save_path: Path) -> None:
        self.principal_components = principal_components
        self.eigenvalues = eigenvalues
        log(INFO, f"principal components shape: {principal_components.shape}")
        log(INFO, f"eigenvalues shape: {eigenvalues.shape}")
        np.save(save_path, principal_components)


class ServerSideMerger:
    def __init__(self, principal_components: NDArrays = [], eigenvalues: NDArrays = []) -> None:
        assert len(principal_components) == len(eigenvalues)
        self.client_eigenvalues = eigenvalues
        self.client_principal_components = principal_components
        self.merged_principal_components: NDArray
        self.merged_eigenvalues: NDArray

    def set_pcs(self, principal_components: NDArrays) -> None:
        self.client_principal_components = principal_components

    def set_eigenvals(self, eigenvalues: NDArrays) -> None:
        self.client_eigenvalues = eigenvalues

    def merge_subspaces(self) -> None:
        eigenvalues_diagonal = [np.diag(eigenvalues_vector) for eigenvalues_vector in self.client_eigenvalues]
        X = [U.T @ S for U, S in zip(self.client_principal_components, eigenvalues_diagonal)]
        svd_input = np.concatenate(X, axis=1)
        new_principal_components, new_eigenvalues, _ = np.linalg.svd(svd_input)
        self.merged_principal_components = new_principal_components
        self.merged_eigenvalues = new_eigenvalues

    def get_principal_components(self) -> Tuple[NDArray, NDArray]:
        return self.merged_principal_components, self.merged_eigenvalues
