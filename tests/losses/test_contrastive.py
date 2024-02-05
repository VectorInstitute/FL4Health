import pytest
import torch

from fl4health.losses.contrastive_loss import ContrastiveLoss

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

global_features = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
local_features = torch.tensor([[[1.0, 2.0], [2.0, 1.0]], [[2.0, 1.0], [1.0, -1.0]]])
previous_local_features = torch.tensor([[[1.0, 2.0], [2.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]])


def test_contrastive_loss() -> None:  # noqa

    # Default temperature is 0.5
    contrastive_loss = ContrastiveLoss(DEVICE)

    contrastive_loss = contrastive_loss(
        features=local_features.reshape(len(local_features), -1),
        positive_pairs=global_features.reshape(1, len(global_features), -1),
        negative_pairs=previous_local_features.reshape(1, len(previous_local_features), -1),
    )

    assert pytest.approx(0.837868, abs=0.0001) == contrastive_loss
