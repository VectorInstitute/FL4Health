import pytest
import torch

from fl4health.losses.perfcl_loss import PerFclLoss


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def test_computing_loss() -> None:
    perfcl_loss_function = PerFclLoss(DEVICE, global_feature_loss_temperature=0.5, local_feature_loss_temperature=0.5)

    local_features = torch.tensor([[1, 1, 1], [1, 1, 1]]).float()
    global_features = torch.tensor([[1, 1, 1], [1, 1, 1]]).float()
    old_local_features = torch.tensor([[0, 0, 0], [0, 0, 0]]).float()
    old_global_features = torch.tensor([[0, 0, 0], [0, 0, 0]]).float()
    initial_global_features = torch.tensor([[1, 1, 1], [1, 1, 1]]).float()

    global_feature_loss, local_feature_loss = perfcl_loss_function(
        local_features,
        old_local_features,
        global_features,
        old_global_features,
        initial_global_features,
    )

    assert pytest.approx(0.126928046, abs=0.00001) == global_feature_loss.item()
    assert pytest.approx(2.126927852, abs=0.00001) == local_feature_loss.item()


def test_perfcl_loss() -> None:
    initial_global_features = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    global_features = torch.tensor([[[1.0, 2.0], [2.0, 1.0]], [[2.0, 1.0], [1.0, -1.0]]])
    old_global_features = torch.tensor([[[1.0, 2.0], [2.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]])
    local_features = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    old_local_features = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])

    perfcl_loss_function = PerFclLoss(DEVICE, global_feature_loss_temperature=0.5, local_feature_loss_temperature=0.5)

    global_feature_loss, _ = perfcl_loss_function(
        local_features.reshape(len(local_features), -1),
        old_local_features.reshape(len(old_local_features), -1),
        global_features.reshape(len(global_features), -1),
        old_global_features.reshape(len(old_global_features), -1),
        initial_global_features.reshape(len(initial_global_features), -1),
    )

    assert pytest.approx(0.837868, abs=0.0001) == global_feature_loss.item()
