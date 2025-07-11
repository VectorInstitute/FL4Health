from logging import INFO, WARNING

import torch
from flwr.common.logger import log
from torch import Tensor, nn
from torch.nn.parameter import Parameter


TWO_D_TENSOR_SHAPE_LENGTH = 2


class PcaModule(nn.Module):
    def __init__(self, low_rank: bool = False, full_svd: bool = False, rank_estimation: int = 6) -> None:
        """
        PyTorch module for performing Principal Component Analysis.

        Notes:
        - If ``low_rank`` is set to True, then a value :math:`q` for ``rank_estimation`` is required (either specified
          by the user or via its default value). If :math:`q` is too far away from the actual rank :math:`k` of the
          data matrix, then the resulting rank-q svd approximation is not guaranteed to be a good approximation of the
          data matrix.
        - If ``low_rank`` is set to True, then a value :math:`q` for ``rank_estimation`` can be chosen according to the
          following criteria:

          - In general, :math:`k \\leq q \\leq \\min(2\\cdot k, m, n)`. For large low-rank matrices, take
            :math:`q = k + l`, where :math:`5 \\leq l \\leq 10`.
            If :math:`k` is relatively small compared to :math:`\\min(m, n)`, choosing :math:`l = 0, 1, 2` may be
            sufficient.
        - If ``low_rank`` is set to True and ``rank_estimation`` is set to :math:`q`, then the module will utilize a
          randomized algorithm to compute a rank-q approximation of the data matrix via SVD.

        For more details on this, see:

        https://pytorch.org/docs/stable/generated/torch.svd_lowrank.html

        and

        https://pytorch.org/docs/stable/generated/torch.pca_lowrank.html

        As per the official documentation of PyTorch, in general, the user should set ``low_rank`` to False. Setting
        it to True would be useful for huge sparse matrices.

        Args:
            low_rank (bool, optional): Indicates whether the data matrix can be well-approximated by a low-rank
                singular value decomposition. If the user has good reasons to believe so, then this parameter can be
                set to True to allow for more efficient computations. Defaults to False.
            full_svd (bool, optional): Indicates whether full SVD or reduced SVD is performed. If ``low_rank`` is set
                to True, then an alternative implementation of SVD will be used and this argument is ignored.
                Defaults to False.
            rank_estimation (int, optional): A slight overestimation of the rank of the data matrix. Only used if
                ``self.low_rank`` is True. Defaults to 6.
        """
        super().__init__()
        self.low_rank = low_rank
        self.full_svd = full_svd
        self.rank_estimation = rank_estimation
        self.principal_components: Parameter
        self.singular_values: Parameter
        self.data_mean: Tensor

    def forward(self, x: Tensor, center_data: bool) -> tuple[Tensor, Tensor]:
        """
        Perform PCA on the data matrix x by computing its SVD.

        **NOTE**: the algorithm assumes that the rows of X are the data points (after reshaping as needed).
        Consequently, the principal components, which are the eigenvectors of X.T @ X, are the right singular vectors
        in the SVD of X.

        Args:
            x (Tensor): Data matrix.
            center_data (bool): If true, then the data mean will be subtracted from all data points prior to
                performing PCA. If ``center_data`` is false, it is expected that the data has already been centered
                and an exception will be thrown if it is not.

        Returns:
            tuple[Tensor, Tensor]: The principal components (i.e., right singular vectors) and their corresponding
            singular values.
        """
        x_prime = self.prepare_data_forward(x, center_data=center_data)
        if self.low_rank:
            log(INFO, "Assuming data matrix is low rank, using low-rank PCA implementation.")
            m, n = x_prime.size(0), x_prime.size(1)
            if self.rank_estimation > m or self.rank_estimation > n:
                log(WARNING, "Estimate of data rank given by user is larger than the actual rank.")
            q = min(self.rank_estimation, m, n)
            _, singular_values, principal_components = torch.pca_lowrank(x_prime, q=q, center=False)
        else:
            if self.full_svd:
                log(INFO, "Performing full SVD on data matrix.")
            else:
                log(INFO, "Performing reduced SVD on data matrix.")
            _, singular_values, vh = torch.linalg.svd(x_prime, full_matrices=self.full_svd)
            principal_components = vh.T
        return principal_components, singular_values

    def maybe_reshape(self, x: Tensor) -> Tensor:
        """
        Reshape input tensor ``X`` as needed so SVD can be computed. Reshaping is required when each data point is an
        N-dimensional tensor because PCA requires ``X`` to be a 2D data matrix.

        Args:
            x (Tensor): Data matrix.

        Returns:
            Tensor: Tensor flattened to be 2D.
        """
        if len(x.size()) == TWO_D_TENSOR_SHAPE_LENGTH:
            return torch.squeeze(x.float())
        dim0 = x.size(0)
        return torch.squeeze(x.view(dim0, -1).float())

    def set_data_mean(self, x: Tensor) -> None:
        """
        The primary purpose of this method is to store the mean of the training data so it can be used to center
        validation/test data later, if needed.

        Args:
            x (Tensor): Data matrix.
        """
        self.data_mean = torch.mean(x, dim=0)

    def center_data(self, x: Tensor) -> Tensor:
        assert self.data_mean is not None
        return x - self.data_mean

    def prepare_data_forward(self, x: Tensor, center_data: bool) -> Tensor:
        """
        Prepare input data ``X`` for PCA by reshaping and centering it as needed.

        Args:
            x (Tensor): Data matrix.
            center_data (bool): If true, then the data mean will be subtracted from all data points prior to
                performing PCA. If ``center_data`` is false, it is expected that the data has already been centered and
                an exception will be thrown if it is not.

        Returns:
            Tensor: Prepared data matrix.
        """
        x = self.maybe_reshape(x)
        if center_data:
            self.set_data_mean(x)
            return self.center_data(x)
        # Check if the mean of X is already (nearly) zero.
        # Throw an exception if it is not.
        data_mean = torch.mean(x, dim=0)
        assert torch.allclose(torch.zeros(data_mean.size()), data_mean, atol=1e-6)
        return x

    def project_lower_dim(self, x: Tensor, k: int | None = None, center_data: bool = False) -> Tensor:
        """
        Project input data ``X`` onto the top k principal components.

        *NOTE**: The result of projection (after centering) is ``X @ U`` because this method assumes that the rows of
        ``X`` are the data points while the columns of ``U`` are the principal components.

        Args:
            x (Tensor): Input data matrix whose rows are the data points.
            k (int | None, optional): The number of principal components onto which projection is done. If ``k`` is
                None, then all principal components will be used in the projection. Defaults to None.
            center_data (bool, optional): If true, then the *training* data mean (learned in the forward pass)
                will be subtracted from all data points prior to projection. If ``center_data`` is false, it is
                expected that the data has already been centered in this manner by the user. Defaults to False.

        Returns:
            Tensor: Projection result.
        """
        x_prime = self.maybe_reshape(x)
        if center_data:
            x_prime = self.center_data(x)
        if k:
            return torch.matmul(x_prime, self.principal_components[:, :k])
        return torch.matmul(x_prime, self.principal_components)

    def project_back(self, x_lower_dim: Tensor, add_mean: bool = False) -> Tensor:
        """
        Project low-dimensional principal representations back into the original space to recover the reconstruction
        of data points.

        Args:
            x_lower_dim (Tensor): Matrix whose rows are low-dimensional principal representations of the original data.
            add_mean (bool, optional): Indicates whether the training data mean should be added to the projection
                result. This can be set to True if the user centered the data prior to dimensionality reduction and
                now wish to add back the data mean. Defaults to False.

        Returns:
            Tensor: Reconstruction of data points.
        """
        x_lower_dim_prime = self.maybe_reshape(x_lower_dim)
        k = x_lower_dim.size(1)
        if add_mean:
            assert self.data_mean is not None
            return torch.matmul(x_lower_dim_prime, self.principal_components[:, :k].T) + self.data_mean
        return torch.matmul(x_lower_dim_prime, self.principal_components[:, :k].T)

    def compute_reconstruction_error(self, x: Tensor, k: int | None, center_data: bool = False) -> float:
        """
        Compute the reconstruction error of X under PCA reconstruction.

        More precisely, if ``X`` is an ``N`` by ``d`` data matrix whose *rows* are the data points,
        and U is the matrix whose *columns* are the principal components of X, then the reconstruction
        loss is defined as ``1 / N * | X @ U @ U.T - X| ** 2``.

        **NOTE**: The reconstruction (after centering) is ``X @ U @ U.T`` because this method assumes that the rows
        of ``X`` are the data points while the columns of U are the principal components.

        Args:
            x (Tensor): Input data tensor whose rows represent data points.
            k (int | None): The number of principal components onto which projection is applied.
            center_data (bool, optional): Indicates whether to subtract data mean prior to projecting the data into a
                lower-dimensional subspace, and whether to add the data mean after projecting back. Defaults to False.

        Returns:
            float: Reconstruction loss as defined above.
        """
        n = x.size(0)
        x_lower_dim = self.project_lower_dim(x, k, center_data=center_data)
        reconstruction = self.project_back(x_lower_dim, add_mean=center_data)
        return (torch.linalg.norm(reconstruction - x) ** 2).item() / n

    def compute_projection_variance(self, x: Tensor, k: int | None, center_data: bool = False) -> float:
        """
        Compute the variance of the data matrix X after projection via PCA.

        The variance is defined as ``| X @ U |_F ** 2``

        Args:
            x (Tensor): input data tensor whose rows represent data points.
            k (int | None): the number of principal components onto which projection is applied.
            center_data (bool, optional): Indicates whether to subtract data mean prior to projecting the data into a
                lower-dimensional subspace, and whether to add the data mean after projecting back. Defaults to False.

        Returns:
            float: Variance after projection as defined above.
        """
        return (torch.linalg.norm(self.project_lower_dim(x, k, center_data)) ** 2).item()

    def compute_cumulative_explained_variance(self) -> float:
        return torch.sum(self.singular_values**2).item()

    def compute_explained_variance_ratios(self) -> Tensor:
        return (self.singular_values**2) / self.compute_cumulative_explained_variance()

    def set_principal_components(self, principal_components: Tensor, singular_values: Tensor) -> None:
        self.principal_components = Parameter(data=principal_components, requires_grad=False)
        self.singular_values = Parameter(data=singular_values, requires_grad=False)
