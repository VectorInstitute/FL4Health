import logging
import re

import pytest
from torch.optim import SGD

from fl4health.utils.nnunet_utils import PolyLRSchedulerWrapper
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
