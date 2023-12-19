from logging import INFO
from typing import Optional, Tuple

import torch
import torch.nn as nn
from flwr.common.logger import log
from torch import Tensor
from torch.nn.parameter import Parameter


class PCAModule(nn.Module):
    def __init__(self, low_rank: bool = True, full_svd: bool = False, rank_estimation: int = 6) -> None:
        """
        PyTorch module for performing Principal Component Analysis.

        Args:
            low_rank (bool, optional): Indicates whether the data matrix is low-rank. If the user has
             prior knowledge that it is, then this parameter can be set to True to allow for more efficient
              computation of SVD. Defaults to True.
            full_svd (bool, optional): Indicates whether full SVD or reduced SVD is performed. Defaults to False.
            rank_estimation (int, optional): An estimation of the rank of the data matrix.
            Only used if self.low_rank is True. Defaults to 6.
        """
        super().__init__()
        self.low_rank = low_rank
        self.full_svd = full_svd
        self.rank_estimation = rank_estimation
        self.principal_components: Parameter
        self.singular_values: Parameter
        self.data_mean: Tensor

    def forward(self, X: Tensor, center_data: bool) -> Tuple[Tensor, Tensor]:
        """
        Perform PCA on the data matrix X by computing its SVD.

        Args:
            X (Tensor): Data matrix.
            center_data (bool): If true, then the data mean will be subtracted from all data points prior to
            performing PCA.

        Returns:
            Tuple[Tensor, Tensor]: The principal components (i.e., left singular vectors) and their corresponding
            singular values.
        """
        X_prime = self.prepare_data_forward(X, center_data=center_data)
        if self.low_rank:
            log(INFO, "Assuming data matrix is low rank, using low-rank PCA implementation.")
            q = min(self.rank_estimation, X_prime.size(0), X_prime.size(1))
            _, S, V = torch.pca_lowrank(X_prime, q=q, center=False)
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

    def maybe_reshape(self, X: Tensor) -> Tensor:
        """
        Reshape input tensor X as needed so SVD can be computed.
        """
        if len(X.size()) == 2:
            return torch.squeeze(X.float())
        else:
            dim0 = X.size(0)
            return torch.squeeze(X.view(dim0, -1).float())

    def set_data_mean(self, X: Tensor) -> None:
        """
        The primary purpose of this method is to store the mean of the
        training data so it can be used to center validation/test data later,
        if needed.
        """
        self.data_mean = torch.mean(X, dim=0)

    def centre_data(self, X: Tensor) -> Tensor:
        assert self.data_mean is not None
        return X - self.data_mean

    def prepare_data_forward(self, X: Tensor, center_data: bool) -> Tensor:
        """
        Prepare input data X for PCA by reshaping and centering it as needed.
        """
        X = self.maybe_reshape(X)
        if center_data:
            self.set_data_mean(X)
            return self.centre_data(X)
        else:
            return X

    def project_lower_dim(self, X: Tensor, k: Optional[int] = None) -> Tensor:
        """
        Project input data X onto the top k principal components.

        Args:
            X (Tensor): Input Data.
            k (Optional[int], optional): The number of principal components
            onto which projection is done. If none, then all principal components will
            be used in the projection. Defaults to None.

        Returns:
            Tensor: Projection result.
        """
        X_prime = self.maybe_reshape(X)
        if k:
            return torch.matmul(X_prime, self.principal_components[:, :k])
        else:
            return torch.matmul(X_prime, self.principal_components)

    def project_back(self, X_lower_dim: Tensor) -> Tensor:
        """
        Project low-dimensional principal representations back into the original space
        to recover the reconstruction of data points.
        """
        X_lower_dim_prime = self.maybe_reshape(X_lower_dim)
        k = X_lower_dim.size(1)
        return torch.matmul(X_lower_dim_prime, self.principal_components[:, :k].T)

    def compute_reconstruction_error(self, X: Tensor, k: Optional[int]) -> float:
        """
        Compute the reconstruction error of X under PCA reconstruction.

        More precisely, if X is n by d, and U is the matrix whose columns are the
        k principal components of X (thus U is d by k), then the reconstruction loss
        is defined as
            1 / n * | X @ U @ U.T - X| ** 2.

        Args:
            X (Tensor): input data tensor whose rows represent data points.
            k (Optional[int]): the number of principal components onto which
                projection is applied.
        Returns:
            float: reconstruction loss as defined above.
        """
        N = X.size(0)
        return (torch.linalg.norm(self.project_back(self.project_lower_dim(X, k)) - X) ** 2).item() / N

    def compute_projection_variance(self, X: Tensor, k: Optional[int]) -> float:
        """
        Compute the variance of the data matrix X after projection via PCA.

        The variance is defined as

        | X @ U |_F ** 2

        Args:
            X (Tensor): input data tensor whose rows represent data points.
            k (Optional[int]): the number of principal components onto which
                projection is applied.
        Returns:
            float: variance after projection as defined above.
        """
        return (torch.linalg.norm(self.project_lower_dim(X, k)) ** 2).item()

    def compute_cumulative_explained_variance(self) -> float:
        return torch.sum(self.singular_values**2).item()

    def compute_explained_variance_ratios(self) -> Tensor:
        return (self.singular_values**2) / self.compute_cumulative_explained_variance()

    def set_principal_components(self, principal_components: Tensor, singular_values: Tensor) -> None:
        self.principal_components = Parameter(data=principal_components, requires_grad=False)
        self.singular_values = Parameter(data=singular_values, requires_grad=False)
