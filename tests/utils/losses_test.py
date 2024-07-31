import pytest
import torch

from fl4health.utils.losses import EvaluationLosses, LossMeter, LossMeterType, TrainingLosses


def test_evaluation_loss_average_meter() -> None:
    losses = [
        EvaluationLosses(
            checkpoint=torch.tensor(1.5837, dtype=torch.float),
            additional_losses={"extra_loss": torch.tensor(2.9562, dtype=torch.float)},
        ),
        EvaluationLosses(
            checkpoint=torch.tensor(2.1562, dtype=torch.float),
            additional_losses={"extra_loss": torch.tensor(6.1582, dtype=torch.float)},
        ),
        EvaluationLosses(
            checkpoint=torch.tensor(0.1290, dtype=torch.float),
            additional_losses={"extra_loss": torch.tensor(0.0429, dtype=torch.float)},
        ),
        EvaluationLosses(
            checkpoint=torch.tensor(7.1020, dtype=torch.float),
            additional_losses={"extra_loss": torch.tensor(5.3532, dtype=torch.float)},
        ),
        EvaluationLosses(
            checkpoint=torch.tensor(4.4930, dtype=torch.float),
            additional_losses={"extra_loss": torch.tensor(7.5267, dtype=torch.float)},
        ),
    ]
    loss_avg_meter = LossMeter[EvaluationLosses](LossMeterType.AVERAGE, EvaluationLosses)
    for loss in losses:
        loss_avg_meter.update(loss)

    computed_losses = loss_avg_meter.compute()
    loss_dict = computed_losses.as_dict()

    assert loss_dict["checkpoint"] == pytest.approx(3.09278, rel=0.01)
    assert loss_dict["extra_loss"] == pytest.approx(4.40744, rel=0.01)


def test_training_loss_average_meter() -> None:
    losses = [
        TrainingLosses(
            backward=torch.tensor(4.3443, dtype=torch.float),
            additional_losses={"extra_loss": torch.tensor(2.9562, dtype=torch.float)},
        ),
        TrainingLosses(
            backward=torch.tensor(3.3837, dtype=torch.float),
            additional_losses={"extra_loss": torch.tensor(6.1582, dtype=torch.float)},
        ),
        TrainingLosses(
            backward=torch.tensor(8.2837, dtype=torch.float),
            additional_losses={"extra_loss": torch.tensor(0.0429, dtype=torch.float)},
        ),
        TrainingLosses(
            backward=torch.tensor(2.5837, dtype=torch.float),
            additional_losses={"extra_loss": torch.tensor(5.3532, dtype=torch.float)},
        ),
        TrainingLosses(
            backward=torch.tensor(0.5837, dtype=torch.float),
            additional_losses={"extra_loss": torch.tensor(7.5267, dtype=torch.float)},
        ),
    ]
    loss_avg_meter = LossMeter[TrainingLosses](LossMeterType.AVERAGE, TrainingLosses)
    for loss in losses:
        loss_avg_meter.update(loss)

    computed_losses = loss_avg_meter.compute()
    loss_dict = computed_losses.as_dict()

    assert loss_dict["backward"] == pytest.approx(3.83582, rel=0.01)
    assert loss_dict["extra_loss"] == pytest.approx(4.40744, rel=0.01)


def test_evaluation_loss_accumulation_meter() -> None:
    losses = [
        EvaluationLosses(
            checkpoint=torch.tensor(1.5837, dtype=torch.float),
            additional_losses={"extra_loss": torch.tensor(2.9562, dtype=torch.float)},
        ),
        EvaluationLosses(
            checkpoint=torch.tensor(2.1562, dtype=torch.float),
            additional_losses={"extra_loss": torch.tensor(6.1582, dtype=torch.float)},
        ),
        EvaluationLosses(
            checkpoint=torch.tensor(0.1290, dtype=torch.float),
            additional_losses={"extra_loss": torch.tensor(0.0429, dtype=torch.float)},
        ),
        EvaluationLosses(
            checkpoint=torch.tensor(7.1020, dtype=torch.float),
            additional_losses={"extra_loss": torch.tensor(5.3532, dtype=torch.float)},
        ),
        EvaluationLosses(
            checkpoint=torch.tensor(4.4930, dtype=torch.float),
            additional_losses={"extra_loss": torch.tensor(7.5267, dtype=torch.float)},
        ),
    ]
    loss_accumulation_meter = LossMeter[EvaluationLosses](LossMeterType.ACCUMULATION, EvaluationLosses)
    for loss in losses:
        loss_accumulation_meter.update(loss)

    computed_losses = loss_accumulation_meter.compute()
    loss_dict = computed_losses.as_dict()

    assert loss_dict["checkpoint"] == pytest.approx(15.4639, rel=0.01)
    assert loss_dict["extra_loss"] == pytest.approx(22.0372, rel=0.01)


def test_training_loss_accumulation_meter() -> None:
    losses = [
        TrainingLosses(
            backward=torch.tensor(4.3443, dtype=torch.float),
            additional_losses={"extra_loss": torch.tensor(2.9562, dtype=torch.float)},
        ),
        TrainingLosses(
            backward=torch.tensor(3.3837, dtype=torch.float),
            additional_losses={"extra_loss": torch.tensor(6.1582, dtype=torch.float)},
        ),
        TrainingLosses(
            backward=torch.tensor(8.2837, dtype=torch.float),
            additional_losses={"extra_loss": torch.tensor(0.0429, dtype=torch.float)},
        ),
        TrainingLosses(
            backward=torch.tensor(2.5837, dtype=torch.float),
            additional_losses={"extra_loss": torch.tensor(5.3532, dtype=torch.float)},
        ),
        TrainingLosses(
            backward=torch.tensor(0.5837, dtype=torch.float),
            additional_losses={"extra_loss": torch.tensor(7.5267, dtype=torch.float)},
        ),
    ]
    loss_accumulation_meter = LossMeter[TrainingLosses](LossMeterType.ACCUMULATION, TrainingLosses)
    for loss in losses:
        loss_accumulation_meter.update(loss)

    computed_losses = loss_accumulation_meter.compute()
    loss_dict = computed_losses.as_dict()

    assert loss_dict["backward"] == pytest.approx(19.1791, rel=0.01)
    assert loss_dict["extra_loss"] == pytest.approx(22.0372, rel=0.01)


def test_training_losses_with_multiple_backward() -> None:
    backward_losses_dict = {
        "model-0": torch.tensor(4.1020, dtype=torch.float),
        "model-1": torch.tensor(6.1020, dtype=torch.float),
        "model-2": torch.tensor(8.1020, dtype=torch.float),
    }
    losses = TrainingLosses(backward=backward_losses_dict)
    losses_dict = losses.as_dict()
    assert losses_dict["model-0"] == pytest.approx(4.1020, rel=0.01)
    assert losses_dict["model-1"] == pytest.approx(6.1020, rel=0.01)
    assert losses_dict["model-2"] == pytest.approx(8.1020, rel=0.01)
    assert len(losses_dict) == 3
