import pytest
import torch

from fl4health.utils.dataset import SslTensorDataset
from fl4health.utils.random import set_all_random_seeds, unset_all_random_seeds


def add_one_transform(t: torch.Tensor) -> torch.Tensor:
    return t + 1.0


def test_ssl_tensor_dataset_construction() -> None:
    set_all_random_seeds(2025)
    data = torch.randn(10, 3, 4)

    dataset = SslTensorDataset(data, None, transform=add_one_transform, target_transform=add_one_transform)

    data_item, target_item = dataset.__getitem__(1)
    assert torch.allclose(data_item, target_item - 1.0, atol=1e-8)

    data_item, target_item = dataset.__getitem__(2)
    assert torch.allclose(data_item, target_item - 1.0, atol=1e-8)

    data_item, target_item = dataset.__getitem__(3)
    assert torch.allclose(data_item, target_item - 1.0, atol=1e-8)

    with pytest.raises(AssertionError):
        dataset = SslTensorDataset(data, data, None, None)

    unset_all_random_seeds()
