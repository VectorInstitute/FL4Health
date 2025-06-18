import pytest
import torch
from flwr.common import Scalar
from torch.utils.data import DataLoader
from tqdm import tqdm

from fl4health.utils.client import (
    check_if_batch_is_empty_and_verify_input,
    clone_and_freeze_model,
    fold_loss_dict_into_metrics,
    maybe_progress_bar,
    move_data_to_device,
    process_and_check_validation_steps,
)
from fl4health.utils.dataset import TensorDataset
from fl4health.utils.logging import LoggingMode
from tests.test_utils.models_for_test import LinearModel


def get_dummy_dataset() -> TensorDataset:
    data = torch.randn(100, 10, 8)
    targets = torch.randint(5, (100,))
    return TensorDataset(data=data, targets=targets)


DUMMY_DATASET = get_dummy_dataset()


def test_process_and_check_validation_steps(caplog: pytest.LogCaptureFixture) -> None:
    dataloader = DataLoader(DUMMY_DATASET, batch_size=15, shuffle=False)

    # No entry in the config, stays None
    num_validation_steps = process_and_check_validation_steps({}, dataloader)
    assert num_validation_steps is None

    # Entry is valid and smaller than the dataloader
    num_validation_steps = process_and_check_validation_steps({"num_validation_steps": 4}, dataloader)

    # Entry is invalid at 0
    with pytest.raises(AssertionError):
        num_validation_steps = process_and_check_validation_steps({"num_validation_steps": 0}, dataloader)

    # Raise warning if too large
    num_validation_steps = process_and_check_validation_steps({"num_validation_steps": 200}, dataloader)
    assert "is larger than the length of the" in caplog.text


def test_fold_loss_dict_into_metrics() -> None:
    metrics: dict[str, Scalar] = {"metric_1": 1.2, "metric_2": 3.2}
    loss = {"loss_1": 1.23, "loss_2": 0.93}
    fold_loss_dict_into_metrics(metrics, loss, LoggingMode.VALIDATION)
    assert metrics["val - loss_1"] == 1.23
    assert metrics["val - loss_2"] == 0.93
    assert metrics["metric_1"] == 1.2
    assert metrics["metric_2"] == 3.2

    metrics = {"metric_1": 1.2, "metric_2": 3.2}
    fold_loss_dict_into_metrics(metrics, loss, LoggingMode.TEST)
    assert metrics["test - loss_1"] == 1.23
    assert metrics["test - loss_2"] == 0.93
    assert metrics["metric_1"] == 1.2
    assert metrics["metric_2"] == 3.2


def test_clone_and_freeze_model() -> None:
    linear_model = LinearModel()
    linear_model.train()
    cloned_model = clone_and_freeze_model(linear_model)

    assert not cloned_model.training
    for param in cloned_model.parameters():
        assert not param.requires_grad

    assert linear_model.training

    for param in linear_model.parameters():
        assert param.requires_grad


def test_check_if_batch_is_empty_and_verify_input() -> None:
    empty_tensor = torch.Tensor([])
    full_tensor = torch.randn((2, 4))
    dict_of_empty_tensors = {"one": torch.Tensor([]), "two": torch.Tensor([])}
    dict_of_same_size_tensors = {"one": torch.randn((2, 4)), "two": torch.randn((2, 4))}
    dict_of_different_size_tensors = {"one": torch.randn((2, 4)), "two": torch.randn((1, 4))}
    bad_input = 1.0

    # Should return false, because not empty
    assert not check_if_batch_is_empty_and_verify_input(full_tensor)
    # Should return false, because not empty
    assert not check_if_batch_is_empty_and_verify_input(dict_of_same_size_tensors)
    # Should return true, because empty
    assert check_if_batch_is_empty_and_verify_input(empty_tensor)
    # Should return true, because empty
    assert check_if_batch_is_empty_and_verify_input(dict_of_empty_tensors)

    # Should throw ValueError because of size mismatch
    with pytest.raises(ValueError):
        check_if_batch_is_empty_and_verify_input(dict_of_different_size_tensors)
    # Should throw TypeError because of bad type
    with pytest.raises(TypeError):
        check_if_batch_is_empty_and_verify_input(bad_input)  # type: ignore


def test_move_data_to_device() -> None:
    full_tensor = torch.randn((2, 4))
    dict_of_same_size_tensors = {"one": torch.randn((2, 4)), "two": torch.randn((2, 4))}
    bad_input = 1.0

    # Best we can do is essentially see that these don't throw errors and return the tensors or dict intact. Mocking
    # actual transfer is sort of hard.
    full_tensor_ = move_data_to_device(full_tensor, torch.device("cpu"))
    assert isinstance(full_tensor_, torch.Tensor)
    assert torch.equal(full_tensor, full_tensor_)
    dict_of_same_size_tensors_ = move_data_to_device(dict_of_same_size_tensors, torch.device("cpu"))
    assert isinstance(dict_of_same_size_tensors_, dict)
    for key, tensor_ in dict_of_same_size_tensors_.items():
        assert torch.equal(dict_of_same_size_tensors[key], tensor_)
    with pytest.raises(TypeError):
        move_data_to_device(bad_input, torch.device("cpu"))  # type: ignore


def test_maybe_progress_bar() -> None:
    iter_without_progress_bar = [0, 1, 2, 3, 4]
    iter_with_progress_bar = maybe_progress_bar(iter_without_progress_bar, display_progress_bar=True)
    assert isinstance(iter_with_progress_bar, tqdm)
    assert not isinstance(iter_without_progress_bar, tqdm)
