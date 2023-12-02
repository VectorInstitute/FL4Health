import mock
import torch

from fl4health.model_bases.ensemble_base import EnsembleAggregationMode, EnsembleModel
from tests.test_utils.models_for_test import SmallCnn


def test_forward_average_mode() -> None:
    models = [SmallCnn(), SmallCnn()]
    em = EnsembleModel(models, EnsembleAggregationMode.AVERAGE)
    data = torch.rand((64, 1, 28, 28))
    ep = em(data)

    assert len(ep) == 3
    assert "ensemble-model-0" in ep and ep["ensemble-model-0"].shape == torch.Size([64, 32])
    assert "ensemble-model-1" in ep and ep["ensemble-model-1"].shape == torch.Size([64, 32])
    assert "ensemble-pred" in ep and ep["ensemble-pred"].shape == torch.Size([64, 32])


def test_forward_vote_mode() -> None:
    models = [SmallCnn(), SmallCnn()]
    em = EnsembleModel(models, EnsembleAggregationMode.VOTE)
    data = torch.rand((64, 1, 28, 28))
    ep = em(data)

    assert len(ep) == 3
    assert "ensemble-model-0" in ep and ep["ensemble-model-0"].shape == torch.Size([64, 32])
    assert "ensemble-model-1" in ep and ep["ensemble-model-1"].shape == torch.Size([64, 32])
    assert "ensemble-pred" in ep and ep["ensemble-pred"].shape == torch.Size([64, 32])


def test_ensemble_vote() -> None:
    fake_instance = mock.Mock()
    tnsr = {
        "ensemble-model-0": torch.tensor([[1.0, 0.5, 0.25], [0.25, 0.33, 0.89], [0.2, 0.4, 0.8]]),
        "ensemble-model-1": torch.tensor([[1.0, 0.5, 0.25], [0.25, 0.33, 0.89], [0.2, 0.4, 0.8]]),
        "ensemble-model-2": torch.tensor([[0.25, 0.5, 1.0], [1.0, 3.21, 0.2], [2.4, 0.5, 0.8]]),
    }
    ensemble_pred = EnsembleModel.ensemble_vote(fake_instance, tnsr)
    gt_ensemble_pred = torch.tensor([[1, 0, 0], [0, 0, 1], [0, 0, 1]])
    assert torch.equal(ensemble_pred, gt_ensemble_pred)
