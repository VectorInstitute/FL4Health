import pytest
import torch

from fl4health.utils.losses import LossAccumulationMeter, LossAverageMeter, Losses


def test_loss_average_meter() -> None:
    losses = [
        Losses(
            checkpoint=torch.tensor(1.5837, dtype=torch.float),
            backward=torch.tensor(4.3443, dtype=torch.float),
            additional_losses={"extra_loss": torch.tensor(2.9562, dtype=torch.float)},
        ),
        Losses(
            checkpoint=torch.tensor(2.1562, dtype=torch.float),
            backward=torch.tensor(3.3837, dtype=torch.float),
            additional_losses={"extra_loss": torch.tensor(6.1582, dtype=torch.float)},
        ),
        Losses(
            checkpoint=torch.tensor(0.1290, dtype=torch.float),
            backward=torch.tensor(8.2837, dtype=torch.float),
            additional_losses={"extra_loss": torch.tensor(0.0429, dtype=torch.float)},
        ),
        Losses(
            checkpoint=torch.tensor(7.1020, dtype=torch.float),
            backward=torch.tensor(2.5837, dtype=torch.float),
            additional_losses={"extra_loss": torch.tensor(5.3532, dtype=torch.float)},
        ),
        Losses(
            checkpoint=torch.tensor(4.4930, dtype=torch.float),
            backward=torch.tensor(0.5837, dtype=torch.float),
            additional_losses={"extra_loss": torch.tensor(7.5267, dtype=torch.float)},
        ),
    ]
    loss_avg_meter = LossAverageMeter()
    for loss in losses:
        loss_avg_meter.update(loss)

    computed_losses = loss_avg_meter.compute()
    loss_dict = computed_losses.as_dict()

    assert loss_dict["checkpoint"] == pytest.approx(3.09278, rel=0.01)
    assert loss_dict["backward"] == pytest.approx(3.83582, rel=0.01)
    assert loss_dict["extra_loss"] == pytest.approx(4.40744, rel=0.01)


def test_loss_accumulation_meter() -> None:
    losses = [
        Losses(
            checkpoint=torch.tensor(1.5837, dtype=torch.float),
            backward=torch.tensor(4.3443, dtype=torch.float),
            additional_losses={"extra_loss": torch.tensor(2.9562, dtype=torch.float)},
        ),
        Losses(
            checkpoint=torch.tensor(2.1562, dtype=torch.float),
            backward=torch.tensor(3.3837, dtype=torch.float),
            additional_losses={"extra_loss": torch.tensor(6.1582, dtype=torch.float)},
        ),
        Losses(
            checkpoint=torch.tensor(0.1290, dtype=torch.float),
            backward=torch.tensor(8.2837, dtype=torch.float),
            additional_losses={"extra_loss": torch.tensor(0.0429, dtype=torch.float)},
        ),
        Losses(
            checkpoint=torch.tensor(7.1020, dtype=torch.float),
            backward=torch.tensor(2.5837, dtype=torch.float),
            additional_losses={"extra_loss": torch.tensor(5.3532, dtype=torch.float)},
        ),
        Losses(
            checkpoint=torch.tensor(4.4930, dtype=torch.float),
            backward=torch.tensor(0.5837, dtype=torch.float),
            additional_losses={"extra_loss": torch.tensor(7.5267, dtype=torch.float)},
        ),
    ]
    loss_accum_meter = LossAccumulationMeter()
    for loss in losses:
        loss_accum_meter.update(loss)

    computed_losses = loss_accum_meter.compute()
    loss_dict = computed_losses.as_dict()

    assert loss_dict["checkpoint"] == pytest.approx(15.4639, rel=0.01)
    assert loss_dict["backward"] == pytest.approx(19.1791, rel=0.01)
    assert loss_dict["extra_loss"] == pytest.approx(22.0372, rel=0.01)
