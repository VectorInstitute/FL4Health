import torch

from fl4health.model_bases.ensemble_base import EnsembleAggregationMode, EnsembleModel
from tests.test_utils.models_for_test import SmallCnn


def test_ensemble_model_average_mode() -> None:
    models = [SmallCnn(), SmallCnn()]
    em = EnsembleModel(models, EnsembleAggregationMode.AVERAGE)
    data = torch.rand((64, 1, 28, 28))
    ep = em(data)

    assert len(ep) == 3
    assert "ensemble-model-0" in ep and ep["ensemble-model-0"].shape == torch.Size([64, 32])
    assert "ensemble-model-1" in ep and ep["ensemble-model-1"].shape == torch.Size([64, 32])
    assert "ensemble-pred" in ep and ep["ensemble-pred"].shape == torch.Size([64, 32])

    gt_ensemble_pred = torch.mean(
        torch.stack([val for key, val in ep.items() if key in ["ensemble-model-0", "ensemble-model-1"]]), dim=0
    )
    assert torch.equal(gt_ensemble_pred, ep["ensemble-pred"])


def test_ensemble_model_vote_mode() -> None:
    models = [SmallCnn(), SmallCnn()]
    em = EnsembleModel(models, EnsembleAggregationMode.VOTE)
    data = torch.rand((64, 1, 28, 28))
    ep = em(data)

    assert len(ep) == 3
    assert "ensemble-model-0" in ep and ep["ensemble-model-0"].shape == torch.Size([64, 32])
    assert "ensemble-model-1" in ep and ep["ensemble-model-1"].shape == torch.Size([64, 32])
    assert "ensemble-pred" in ep and ep["ensemble-pred"].shape == torch.Size([64, 32])
