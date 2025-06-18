import math
from collections.abc import Generator

import pytest
import torch
from torch import Tensor
from torch.nn.parameter import Parameter

from fl4health.model_bases.pca import PcaModule
from fl4health.utils.random import set_all_random_seeds, unset_all_random_seeds


data_dimension = 256
N = 2048
N_small = 128
seed = 83
small_rank = 16


# For more information of this type hint,
# see https://docs.python.org/3.8/library/typing.html#typing.Generator/
@pytest.fixture
def setup_random_seeds() -> Generator[None, None, None]:
    set_all_random_seeds(seed)
    yield
    unset_all_random_seeds()


def create_full_rank_data() -> Tensor:
    # Create a full-rank data matrix of size (N, data_dimension).
    # Since torch.rand() is very likely to return a full-rank matrix,
    # we use it to achieve this (after setting seed).
    x = torch.rand(N, data_dimension)
    assert torch.linalg.matrix_rank(x) == data_dimension
    return x


def create_low_rank_data() -> Tensor:
    # Create a low-rank data matrix of size (N, data_dimension) and rank = low_rank.
    a = torch.rand(small_rank - 1, data_dimension)
    assert torch.linalg.matrix_rank(a) == small_rank - 1
    b = torch.ones(N - small_rank + 1, data_dimension)
    x = torch.concat((a, b), dim=0)
    assert torch.linalg.matrix_rank(x) == small_rank
    return x


def test_reshaping() -> None:
    pca_module = PcaModule()

    # Test that data with more than 2 dimensions are properly reshaped by the pca module.
    a = torch.rand(3, 4, 5)
    a_prime = pca_module.maybe_reshape(a)
    assert a_prime.size() == torch.Size([3, 20])
    assert (a_prime.view(3, 4, 5) == a).all()

    # Test that data with 2 dimensions are left unchanged.
    b = torch.rand(5, 6)
    b_prime = pca_module.maybe_reshape(b)
    assert (b_prime == b).all()


def test_centering(setup_random_seeds: Generator[None, None, None]) -> None:
    # Test that the pca module properly centers the data matrix.
    x = create_full_rank_data()
    data_mean = torch.mean(x, dim=0)
    pca_module = PcaModule()
    x_prime = pca_module.prepare_data_forward(x, center_data=True)
    # After centering, the data matrix should have zero mean.
    new_mean = torch.mean(x_prime, dim=0)
    assert torch.allclose(torch.zeros(new_mean.size()), new_mean, atol=1e-7)
    assert (pca_module.data_mean == data_mean).all()

    x = x - data_mean
    x_prime = pca_module.prepare_data_forward(x, center_data=False)
    assert torch.allclose(x_prime, x, atol=1e-7)


def test_full_svd_full_rank_data(setup_random_seeds: Generator[None, None, None]) -> None:
    pca_module = PcaModule(low_rank=False, full_svd=True)
    x = create_full_rank_data()
    principal_components, singular_values = pca_module(x, center_data=True)
    assert principal_components.size(1) == data_dimension and len(singular_values) == data_dimension
    pca_module.set_principal_components(principal_components, singular_values)
    # Since X has rank = data_dimension and the number of principal components
    # used in reconstruction equals this rank (since we perform full svd),
    # we should expect reconstruction_error to be close to zero.
    reconstruction_error = pca_module.compute_reconstruction_error(x, k=None, center_data=True)
    assert math.isclose(reconstruction_error, 0.0, abs_tol=1e-8)


def test_full_svd_low_rank_data(setup_random_seeds: Generator[None, None, None]) -> None:
    pca_module = PcaModule(low_rank=False, full_svd=True)
    x = create_low_rank_data()
    principal_components, singular_values = pca_module(x, center_data=True)
    assert principal_components.size(1) == data_dimension and len(singular_values) == data_dimension
    # Since X is low-rank, it is expected that only the first small_rank singular values should be nonzero.
    # The remaining singular values should be (nearly) zero.
    assert torch.allclose(singular_values[small_rank:], torch.zeros(data_dimension - small_rank), atol=5e-4)
    pca_module.set_principal_components(principal_components, singular_values)

    # Since X has rank = small_rank and this is also the number of principal components
    # used in reconstruction, we should expect reconstruction_error to be close to zero.
    reconstruction_error = pca_module.compute_reconstruction_error(x, k=small_rank, center_data=True)
    assert math.isclose(reconstruction_error, 0.0, abs_tol=1e-8)


def test_reduced_svd_full_rank_data(setup_random_seeds: Generator[None, None, None]) -> None:
    pca_module = PcaModule(low_rank=False, full_svd=False)
    x = torch.rand(N_small, data_dimension)
    principal_components, singular_values = pca_module(x, center_data=True)
    assert principal_components.size(1) == N_small and len(singular_values) == N_small
    # We should still expect (nearly) perfect reconstruction.
    pca_module.set_principal_components(principal_components, singular_values)
    reconstruction_error = pca_module.compute_reconstruction_error(x, k=None, center_data=True)
    assert math.isclose(reconstruction_error, 0.0, abs_tol=1e-8)


def test_reduced_svd_low_rank_data(setup_random_seeds: Generator[None, None, None]) -> None:
    pca_module = PcaModule(low_rank=False, full_svd=False)
    x = create_low_rank_data()
    principal_components, singular_values = pca_module(x, center_data=True)
    assert principal_components.size(1) == data_dimension and len(singular_values) == data_dimension
    assert torch.allclose(singular_values[small_rank:], torch.zeros(data_dimension - small_rank), atol=5e-4)
    # We should still expect (nearly) perfect reconstruction.
    pca_module.set_principal_components(principal_components, singular_values)
    reconstruction_error = pca_module.compute_reconstruction_error(x, k=small_rank, center_data=True)
    assert math.isclose(reconstruction_error, 0.0, abs_tol=1e-8)


def test_low_rank_svd(setup_random_seeds: Generator[None, None, None]) -> None:
    # Following the same approach, we test the low_rank svd functionality.
    q = small_rank + 4
    pca_module = PcaModule(low_rank=True, full_svd=False, rank_estimation=q)
    x = create_low_rank_data()
    principal_components, singular_values = pca_module(x, center_data=True)
    assert principal_components.size(1) == q and len(singular_values) == q
    assert torch.allclose(singular_values[small_rank:], torch.zeros(q - small_rank), atol=5e-4)
    pca_module.set_principal_components(principal_components, singular_values)
    reconstruction_error = pca_module.compute_reconstruction_error(x, k=small_rank, center_data=True)
    assert math.isclose(reconstruction_error, 0.0, abs_tol=1e-8)


def test_compute_explained_variance_ratios() -> None:
    pca_module = PcaModule()
    pca_module.singular_values = Parameter(torch.Tensor([2.0, 3.0, 4.0, 5.0]), requires_grad=False)
    ratios = pca_module.compute_explained_variance_ratios()
    target_ratios = torch.Tensor([4.0, 9.0, 16.0, 25.0]) / 54.0
    assert torch.allclose(ratios, target_ratios, atol=1e-7)
