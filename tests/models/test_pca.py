import torch
from torch import Tensor

from fl4health.model_bases.pca import PcaModule
from fl4health.utils.random import set_all_random_seeds

data_dimension = 256
N = 2048
seed = 83
small_rank = 16
set_all_random_seeds(seed)


def create_full_rank_data() -> Tensor:
    # Create a full-rank data matrix of size (N, data_dimension).
    # Since torch.rand() is very likely to return a full-rank matrix,
    # we use it to achieve this (after setting seed).
    X = torch.rand(N, data_dimension)
    assert torch.linalg.matrix_rank(X) == data_dimension
    return X


def create_low_rank_data() -> Tensor:
    # Create a low-rank data matrix of size (N, data_dimension) and rank = low_rank.
    A = torch.rand(small_rank - 1, data_dimension)
    assert torch.linalg.matrix_rank(A) == small_rank - 1
    B = torch.ones(N - small_rank + 1, data_dimension)
    X = torch.concat((A, B), dim=0)
    assert torch.linalg.matrix_rank(X) == small_rank
    return X


def test_reshaping() -> None:
    pca_module = PcaModule()

    # Test that data with more than 2 dimensions are properly reshaped by the pca module.
    A = torch.rand(3, 4, 5)
    A_prime = pca_module.maybe_reshape(A)
    assert A_prime.size() == torch.Size([3, 20])
    assert (A_prime.view(3, 4, 5) == A).all()

    # Test that data with 2 dimensions are left unchanged.
    B = torch.rand(5, 6)
    B_prime = pca_module.maybe_reshape(B)
    assert (B == B_prime).all()


def test_centering() -> None:
    # Test that the pca module properly centers the data matrix.
    X = create_full_rank_data()
    data_mean = torch.mean(X, dim=0)
    pca_module = PcaModule()
    X_prime = pca_module.prepare_data_forward(X, center_data=True)
    # After centering, the data matrix should have zero mean.
    new_mean = torch.mean(X_prime, dim=0)
    assert torch.allclose(torch.zeros(new_mean.size()), new_mean, atol=1e-7)
    assert (pca_module.data_mean == data_mean).all()


def test_full_svd() -> None:
    pca_module = PcaModule(low_rank=False, full_svd=True)
    # Since X has rank = data_dimension and the number of principal components
    # used in reconstruction equals this rank (since we perform full svd),
    # we should expect perfect reconstruction.
    X = create_full_rank_data()
    principal_components, singular_values = pca_module(X, center_data=True)
    pca_module.set_principal_components(principal_components, singular_values)
    reconstruction_error = pca_module.compute_reconstruction_error(X, k=None, center_data=True)
    assert abs(reconstruction_error) < 1e-10


def test_reduced_svd() -> None:
    pca_module = PcaModule(low_rank=False, full_svd=False)
    X = torch.rand(small_rank, data_dimension)
    principal_components, singular_values = pca_module(X, center_data=True)
    assert principal_components.size(1) == small_rank and len(singular_values) == small_rank
    # We should still expect perfect reconstruction.
    pca_module.set_principal_components(principal_components, singular_values)
    reconstruction_error = pca_module.compute_reconstruction_error(X, k=None, center_data=True)
    assert abs(reconstruction_error) < 1e-10


def test_low_rank_svd() -> None:
    q = small_rank + 4
    pca_module = PcaModule(low_rank=True, full_svd=False, rank_estimation=q)
    X = create_low_rank_data()
    principal_components, singular_values = pca_module(X, center_data=True)
    assert principal_components.size(1) == q and len(singular_values) == q
    pca_module.set_principal_components(principal_components, singular_values)
    reconstruction_error = pca_module.compute_reconstruction_error(X, k=small_rank, center_data=True)
    assert abs(reconstruction_error) < 1e-10
