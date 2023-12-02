from typing import Dict

import mock
import torch
import torch.nn as nn

from fl4health.model_bases.ensemble_base import EnsembleAggregationMode, EnsembleModel
from tests.test_utils.models_for_test import SmallCnn


def test_forward_average_mode() -> None:
    models: Dict[str, nn.Module] = {"ensemble-model-0": SmallCnn(), "ensemble-model-1": SmallCnn()}
    em = EnsembleModel(models, EnsembleAggregationMode.AVERAGE)
    data = torch.rand((64, 1, 28, 28))
    ep = em(data)

    assert len(ep) == 3
    assert "ensemble-model-0" in ep and ep["ensemble-model-0"].shape == torch.Size([64, 32])
    assert "ensemble-model-1" in ep and ep["ensemble-model-1"].shape == torch.Size([64, 32])
    assert "ensemble-pred" in ep and ep["ensemble-pred"].shape == torch.Size([64, 32])


def test_forward_vote_mode() -> None:
    models: Dict[str, nn.Module] = {"ensemble-model-0": SmallCnn(), "ensemble-model-1": SmallCnn()}
    em = EnsembleModel(models, EnsembleAggregationMode.VOTE)
    data = torch.rand((64, 1, 28, 28))
    ep = em(data)

    assert len(ep) == 3
    assert "ensemble-model-0" in ep and ep["ensemble-model-0"].shape == torch.Size([64, 32])
    assert "ensemble-model-1" in ep and ep["ensemble-model-1"].shape == torch.Size([64, 32])
    assert "ensemble-pred" in ep and ep["ensemble-pred"].shape == torch.Size([64, 32])


def test_ensemble_vote() -> None:
    fake_instance = mock.Mock()
    tnsr_list = [
        torch.tensor([[1.0, 0.5, 0.25], [0.25, 0.33, 0.89], [0.2, 0.4, 0.8]]),
        torch.tensor([[1.0, 0.5, 0.25], [0.25, 0.33, 0.89], [0.2, 0.4, 0.8]]),
        torch.tensor([[0.25, 0.5, 1.0], [1.0, 3.21, 0.2], [2.4, 0.5, 0.8]]),
    ]
    ensemble_pred = EnsembleModel.ensemble_vote(fake_instance, tnsr_list)
    gt_ensemble_pred = torch.tensor([[1, 0, 0], [0, 0, 1], [0, 0, 1]])
    assert torch.equal(ensemble_pred, gt_ensemble_pred)

    tnsr_list2 = [
        torch.tensor([[[1.0, 0.5, 0.25], [0.25, 0.33, 0.89], [0.2, 0.4, 0.8]] for _ in range(3)]),
        torch.tensor([[[1.0, 0.5, 0.25], [0.25, 0.33, 0.89], [0.2, 0.4, 0.8]] for _ in range(3)]),
        torch.tensor([[[0.25, 0.5, 1.0], [1.0, 3.21, 0.2], [2.4, 0.5, 0.8]] for _ in range(3)]),
    ]

    ensemble_pred2 = EnsembleModel.ensemble_vote(fake_instance, tnsr_list2)
    gt_ensemble_pred2 = torch.tensor([[[1, 0, 0], [0, 0, 1], [0, 0, 1]] for _ in range(3)])
    assert torch.equal(ensemble_pred2, gt_ensemble_pred2)


def test_ensenble_average() -> None:
    fake_instance = mock.Mock()
    tnsr_list = [torch.eye(5) * (i + 1.0) for i in range(3)]
    ensemble_pred = EnsembleModel.ensemble_average(fake_instance, tnsr_list)
    gt_ensemble_pred = torch.eye(5) * 2.0
    assert torch.equal(ensemble_pred, gt_ensemble_pred)

    tnsr_list2 = [torch.ones((7, 7, 7)), torch.zeros((7, 7, 7)), torch.ones((7, 7, 7)), torch.zeros((7, 7, 7))]
    ensemble_pred2 = EnsembleModel.ensemble_average(fake_instance, tnsr_list2)
    gt_ensemble_pred2 = torch.ones((7, 7, 7)) * 0.5
    assert torch.equal(ensemble_pred2, gt_ensemble_pred2)
