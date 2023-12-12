from logging import INFO
from typing import Mapping, Optional, Tuple

import torch
import torch.nn as nn
from flwr.common.logger import log
from torch import Tensor
from torch.nn.parameter import Parameter


class PCAModule(nn.Module):
    def __init__(self, low_rank: bool = True, full_svd: bool = False, rank_estimation: int = 6) -> None:
        super().__init__()
        self.low_rank = low_rank
        self.full_svd = full_svd
        self.rank_estimation = rank_estimation
        self.principal_components: Parameter
        self.singular_values: Parameter

    def forward(self, X: Tensor, center: bool = True) -> Tuple[Tensor, Tensor]:
        X_prime = self.prepare_data(X, center=center)
        if self.low_rank:
            log(INFO, "Assuming data matrix is low rank, using low-rank PCA implementation.")
            q = min(self.rank_estimation, X_prime.size(0), X_prime.size(1))
            _, S, V = torch.pca_lowrank(X_prime, q=q, center=center)
            principal_components = V
        else:
            if self.full_svd:
                log(INFO, "Performing full SVD on data matrix.")
            else:
                log(INFO, "Performing reduced SVD on data matrix.")
            _, S, Vh = torch.linalg.svd(X_prime, full_matrices=self.full_svd)
            principal_components = Vh.T
        singular_values = S
        return principal_components, singular_values

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

    def project_lower_dim(self, X: Tensor, k: Optional[int] = None) -> Tensor:
        X_prime = self._maybe_reshape(X)
        if k:
            return torch.matmul(X_prime, self.principal_components[:, :k])
        else:
            return torch.matmul(X_prime, self.principal_components)

    def project_back(self, X_lower_dim: Tensor, k: Optional[int] = None) -> Tensor:
        X_lower_dim_prime = self._maybe_reshape(X_lower_dim)
        if k:
            return torch.matmul(X_lower_dim_prime, self.principal_components[:, :k].T)
        else:
            return torch.matmul(X_lower_dim_prime, self.principal_components.T)

    def compute_reconstruction_loss(self, X: Tensor, k: Optional[int]) -> float:
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
        return torch.linalg.norm(self.project_back(self.project_lower_dim(X, k), k) - X)

    def compute_cumulative_explained_variance(self) -> float:
        return torch.sum(self.singular_values**2).item()

    def compute_explained_variance_ratios(self) -> Tensor:
        return (self.singular_values**2) / self.compute_cumulative_explained_variance()

    def set_principal_components(self, principal_components: Tensor, singular_values: Tensor) -> None:
        self.principal_components = Parameter(data=principal_components, requires_grad=False)
        self.singular_values = Parameter(data=singular_values, requires_grad=False)

    def load_state_dict(self, state_dict: Mapping[str, Tensor], strict: bool = True) -> None:
        principal_components = state_dict["principal_components"]
        singular_values = state_dict["singular_values"]
        self.set_principal_components(principal_components, singular_values)
