import logging
import re

import pytest
import torch
from torch.optim import SGD

from fl4health.utils.nnunet_utils import PolyLRSchedulerWrapper, get_dataset_n_voxels, prepare_loss_arg
from tests.test_utils.models_for_test import MnistNetWithBnAndFrozen


def test_poly_lr_scheduler(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING)
    pattern = r"Current LR step of \d+ reached Max Steps of \d+. LR will remain fixed."

    max_steps = 100
    exponent = 1
    steps_per_lr = 10
    initial_lr = 0.5

    model = MnistNetWithBnAndFrozen()
    opt = SGD(model.parameters(), lr=initial_lr)
    lr_scheduler = PolyLRSchedulerWrapper(
        optimizer=opt, max_steps=max_steps, initial_lr=initial_lr, exponent=exponent, steps_per_lr=steps_per_lr
    )

    assert lr_scheduler.num_windows == 10.0
    assert lr_scheduler.initial_lr == initial_lr

    prev_lr = None
    for step in range(max_steps):
        curr_lr = lr_scheduler.get_lr()[0]

        if step % steps_per_lr == 0:
            assert curr_lr != prev_lr
        else:
            assert curr_lr == prev_lr

        prev_lr = curr_lr

        lr_scheduler.step()

        assert not re.search(pattern, caplog.text)

    lr_scheduler.step()

    assert re.search(pattern, caplog.text)


def test_prepare_loss_arg() -> None:
    pure_tensor = torch.randn((3, 4, 2))
    single_tensor_dict = {"primary": pure_tensor}
    list_of_tensors = [pure_tensor, torch.randn((3, 2))]

    # If just a tensor, no change should be made
    same_tensor = prepare_loss_arg(pure_tensor)
    assert isinstance(same_tensor, torch.Tensor)
    torch.equal(pure_tensor, same_tensor)

    # If the dictionary has just one tensor, then flatten it to a pure tensor
    same_tensor = prepare_loss_arg(single_tensor_dict)
    assert isinstance(same_tensor, torch.Tensor)
    torch.equal(pure_tensor, same_tensor)

    # Make sure we throw an error if the type is unrecognized
    with pytest.raises(ValueError):
        prepare_loss_arg(list_of_tensors)  # type: ignore


def test_get_dataset_n_voxels() -> None:
    mock_source_plans_3d_full_res = {"configurations": {"3d_fullres": {"median_image_size_in_voxels": 100}}}

    mock_source_plans_2d = {"configurations": {"2d": {"median_image_size_in_voxels": 200}}}

    assert get_dataset_n_voxels(mock_source_plans_3d_full_res, 10) == 1000

    assert get_dataset_n_voxels(mock_source_plans_2d, 10) == 2000
