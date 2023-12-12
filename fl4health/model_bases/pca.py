from logging import INFO
from typing import Mapping, Tuple

import torch
import torch.nn as nn
from flwr.common.logger import log
from torch import Tensor
from torch.nn.parameter import Parameter


class PCAModule(nn.Module):
    def __init__(self, num_components: int, low_rank: bool = True, rank_estimation: int = 6) -> None:
        super().__init__()
        self.num_components = num_components
        self.low_rank = low_rank
        self.rank_estimation = rank_estimation
        self.principal_components: Parameter
        self.eigenvalues: Parameter

    def forward(self, X: Tensor, center: bool = True) -> Tuple[Tensor, Tensor]:
        X_prime = self.prepare_data(X, center=center)
        if self.low_rank:
            log(INFO, "Assuming data matrix is low rank, using low-rank PCA implementation.")
            q = min(self.rank_estimation, X_prime.size(0), X_prime.size(1))
            _, S, V = torch.pca_lowrank(X_prime, q=q, center=center)
            principal_components = V
        else:
            log(INFO, "Performing full SVD on data matrix.")
            _, S, Vh = torch.linalg.svd(X_prime, full_matrices=True)
            principal_components = Vh.T
        # Since singular values are the square roots
        eigenvalues = (S**2)[: self.num_components]
        pc_result = principal_components[:, : self.num_components]
        return pc_result, eigenvalues

    def _maybe_reshape(self, X: Tensor) -> Tensor:
        if len(X.size()) == 2:
            return X.float()
        else:
            dim0 = X.size(0)
            return X.view((dim0, -1)).float()

    def _centre_data(self, X: Tensor) -> Tensor:
        column_mean = torch.mean(X, dim=1)
        return (X.T - column_mean).T

    def prepare_data(self, X: Tensor, center: bool) -> Tensor:
        X = self._maybe_reshape(X)
        if center:
            X = self._centre_data(X)
        return X

    def project_lower_dim(self, X: Tensor) -> Tensor:
        X_prime = self._maybe_reshape(X)
        return torch.matmul(X_prime, self.principal_components)

    def project_back(self, X_lower_dim: Tensor) -> Tensor:
        X_lower_dim_prime = self._maybe_reshape(X_lower_dim)
        return torch.matmul(X_lower_dim_prime, self.principal_components.T)

    def compute_reconstruction_loss(self, X: Tensor) -> float:
        """
        Compute the reconstruction loss of X under PCA reconstruction.

        More precisely, if X is n by d, and U is the matrix whose columns are the
        k principal components of X (thus U is d by k), then the reconstruction loss
        is defined as
            | X @ U @ U.T - X| ** 2

        Args:
            X (Tensor): input tensor

        Returns:
            float: reconstruction loss as defined above.
        """
        return torch.linalg.norm(self.project_back(self.project_lower_dim(X)) - X)

    def compute_cumulative_explained_variance(self) -> float:
        return torch.sum(self.eigenvalues).item()

    def compute_explained_variance_ratios(self) -> Tensor:
        return self.eigenvalues / self.compute_cumulative_explained_variance()

    def set_principal_components(self, principal_components: Tensor, eigenvalues: Tensor) -> None:
        self.principal_components = Parameter(data=principal_components, requires_grad=False)
        self.eigenvalues = Parameter(data=eigenvalues, requires_grad=False)

    def load_state_dict(self, state_dict: Mapping[str, Tensor], strict: bool = True) -> None:
        principal_components = state_dict["principal_components"]
        eigenvalues = state_dict["eigenvalues"]
        self.set_principal_components(principal_components, eigenvalues)
