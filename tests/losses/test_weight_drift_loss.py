import pytest
import torch

from fl4health.losses.weight_drift_loss import WeightDriftLoss
from tests.test_utils.models_for_test import SmallCnn


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def test_contrastive_loss() -> None:
    torch.manual_seed(42)
    model = SmallCnn()

    perturbed_tensors = [layer_weights.clone().detach() + 0.1 for layer_weights in model.state_dict().values()]

    weight_drift_loss = WeightDriftLoss(DEVICE)

    loss_value = weight_drift_loss(model, perturbed_tensors, 2.0)

    assert pytest.approx(loss_value.detach().item(), abs=0.002) == (1.5 + 0.06 + 24.0 + 81.92 + 0.16 + 0.32)
