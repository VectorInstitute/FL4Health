from logging import INFO
from pathlib import Path
from typing import Tuple

import numpy as np
from flwr.common.logger import log
from flwr.common.typing import NDArray, NDArrays
from sklearn.decomposition import PCA


class ClientSidePCA:
    def __init__(self, n_components: int) -> None:
        self._pca = PCA(n_components=n_components, random_state=42)
        self.principal_components: NDArray
        self.eigenvalues: NDArray

    def compute_pc(self, X: NDArray) -> Tuple[NDArray, NDArray]:
        self._pca.fit(X)
        components, eigen_vals = self._pca.components_, self._pca.explained_variance_
        return components, eigen_vals

    def update_pcs(self, pcs: NDArray, eigenvalues: NDArray) -> None:
        self.principal_components = pcs
        self.eigenvalues = eigenvalues

    def get_pcs(self) -> Tuple[NDArray, NDArray]:
        return self.principal_components, self.eigenvalues

    def pc_projection(self, X: NDArray) -> NDArray:
        return self.principal_components @ X

    def save_pcs(self, pc_path: Path) -> None:
        pcs, eigenvalues = self.get_pcs()
        log(INFO, f"principal components shape: {pcs.shape}")
        log(INFO, f"eigenvalues shape: {eigenvalues.shape}")
        np.save(pc_path, pcs)


class ServerSideMerger:
    def __init__(self, pcs: NDArrays = [], eigenvals: NDArrays = []) -> None:
        assert len(pcs) == len(eigenvals)
        self.client_eigenvals = eigenvals
        self.client_pcs = pcs
        self.merged_pcs: NDArray
        self.merged_eigen_vals: NDArray

    def set_pcs(self, pcs: NDArrays) -> None:
        self.client_pcs = pcs

    def set_eigenvals(self, eigenvals: NDArrays) -> None:
        self.client_eigenvals = eigenvals

    def merge_subspaces(self) -> None:
        eigen_vals_diag = [np.diag(e_val_vec) for e_val_vec in self.client_eigenvals]
        X = [U.T @ S for U, S in zip(self.client_pcs, eigen_vals_diag)]
        svd_input = np.concatenate(X, axis=1)
        new_pcs, new_eigen_vals, _ = np.linalg.svd(svd_input)
        self.merged_pcs = new_pcs
        self.merged_eigen_vals = new_eigen_vals

    def get_principal_components(self) -> Tuple[NDArray, NDArray]:
        return self.merged_pcs, self.merged_eigen_vals
