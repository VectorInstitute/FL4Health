import pytest
import torch

from fl4health.utils.dataset import DictionaryDataset, SslTensorDataset, select_by_indices
from fl4health.utils.random import set_all_random_seeds, unset_all_random_seeds


def construct_dictionary_dataset() -> DictionaryDataset:
    # set seed for creation
    set_all_random_seeds(2025)
    random_inputs = [torch.randn((3)) for _ in range(10)]
    # Note high is exclusive here
    random_labels = torch.randint(low=0, high=5, size=(10, 1))

    # unset seed thereafter
    unset_all_random_seeds()
    return DictionaryDataset({"primary": random_inputs}, random_labels)


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


def test_select_by_indices() -> None:
    dictionary_dataset = construct_dictionary_dataset()
    dictionary_dataset_subset = select_by_indices(dictionary_dataset, torch.Tensor([0, 1, 5]).int())

    data_0 = dictionary_dataset_subset.data["primary"][0]
    data_1 = dictionary_dataset_subset.data["primary"][1]
    data_5 = dictionary_dataset_subset.data["primary"][2]

    assert torch.allclose(data_0, torch.Tensor([-0.8716, 0.1114, 1.2044]), atol=1e-4)
    assert torch.allclose(data_1, torch.Tensor([-0.1803, 1.0021, 0.7914]), atol=1e-4)
    assert torch.allclose(data_5, torch.Tensor([0.0521, 1.1838, 1.6855]), atol=1e-4)

    with pytest.raises(TypeError):
        # Intentionally providing a bad type
        select_by_indices(torch.Tensor([0.1, 0.2]), torch.Tensor([0, 1, 5]))  # type: ignore
