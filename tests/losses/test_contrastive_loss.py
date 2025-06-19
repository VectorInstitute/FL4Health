import pytest
import torch
import torch.nn.functional as F

from fl4health.losses.contrastive_loss import MoonContrastiveLoss, NtXentLoss


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def nt_xent(x1: torch.Tensor, x2: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    """
    Fed-X Alternative Implementation of NT-Xent to Compare against.
    https://github.com/Sungwon-Han/FEDX/blob/main/losses.py#L11.
    """
    x1 = F.normalize(x1, dim=1)
    x2 = F.normalize(x2, dim=1)
    batch_size = x1.size(0)
    out = torch.cat([x1, x2], dim=0)
    sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / t)
    mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()
    sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)
    pos_sim = torch.exp(torch.sum(x1 * x2, dim=-1) / t)
    pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
    return (-torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()


def test_moon_contrastive_loss() -> None:
    # Default temperature is 0.5
    contrastive_loss = MoonContrastiveLoss(DEVICE)

    global_features = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    local_features = torch.tensor([[[1.0, 2.0], [2.0, 1.0]], [[2.0, 1.0], [1.0, -1.0]]])
    previous_local_features = torch.tensor([[[1.0, 2.0], [2.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]])

    loss_value = contrastive_loss(
        features=local_features.reshape(len(local_features), -1),
        positive_pairs=global_features.reshape(1, len(global_features), -1),
        negative_pairs=previous_local_features.reshape(1, len(previous_local_features), -1),
    ).item()

    assert pytest.approx(0.837868, abs=0.0001) == loss_value

    features = torch.tensor([[1.0, 1.0, 1.0]]).float()
    positive_pairs = torch.tensor([[[1.0, 1.0, 1.0]]]).float()
    negative_pairs = torch.tensor([[[0.0, 0.0, 0.0]]]).float()
    loss_value = contrastive_loss(features, positive_pairs, negative_pairs).item()

    assert loss_value == pytest.approx(0.1269, rel=0.01)

    features = torch.tensor([[0.0, 0.0, 0.0]]).float()
    positive_pairs = torch.tensor([[[1.0, 1.0, 1.0]]]).float()
    negative_pairs = torch.tensor([[[0.0, 0.0, 0.0]]]).float()
    loss_value = contrastive_loss(features, positive_pairs, negative_pairs).item()

    assert loss_value == pytest.approx(0.6931, rel=0.01)


def test_compute_negative_similarities() -> None:
    contrastive_loss = MoonContrastiveLoss(DEVICE)
    features = torch.tensor([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]])
    negative_pairs = torch.Tensor(
        [[[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]], [[-1.0, -1.0, -1.0], [1.0, 0.0, 0.0]], [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]
    )

    similarities = contrastive_loss.compute_negative_similarities(features, negative_pairs)

    assert similarities.shape == torch.Size([3, 2])
    assert torch.allclose(similarities[0], torch.Tensor([1.0, 0.0]), atol=1e-5)
    assert torch.allclose(similarities[1], torch.Tensor([-1.0, 0.0]), atol=1e-5)
    target_sim = 1.0 / torch.norm(torch.tensor([1.0, 1.0, 1.0])).item()
    assert torch.allclose(similarities[2], torch.Tensor([target_sim, 0.0]), atol=1e-5)

    # Test that when shapes mismatch, the computation fails.
    features = torch.rand((4, 3))
    negative_pairs_1 = torch.rand((5, 3, 3))
    negative_pairs_2 = torch.rand((5, 4, 2))

    # mismatch in batch dimension
    with pytest.raises(AssertionError):
        contrastive_loss.compute_negative_similarities(features, negative_pairs_1)

    # mismatch in feature dimension
    with pytest.raises(AssertionError):
        contrastive_loss.compute_negative_similarities(features, negative_pairs_2)


def test_contrastive_loss() -> None:
    contrastive_loss = NtXentLoss(DEVICE, temperature=0.5)

    features = torch.stack([torch.arange(1, 11) for _ in range(10)]).float()
    transformed_features = torch.stack([torch.arange(10, 101, 10) for _ in range(10)]).T.float()

    result = contrastive_loss(features, transformed_features).item()
    expected_result = nt_xent(features, transformed_features).item()

    assert result == pytest.approx(expected_result, rel=0.01)

    contrastive_loss2 = NtXentLoss(DEVICE, temperature=0.1)

    features2 = torch.stack([torch.arange(0, 41, 2) for _ in range(10)]).float()
    transformed_features2 = torch.stack([torch.arange(0, 101, 5) for _ in range(10)]).float()

    result2 = contrastive_loss2(features2, transformed_features2).item()
    expected_result2 = nt_xent(features2, transformed_features2, t=0.1).item()

    assert result2 == pytest.approx(expected_result2, rel=0.01)

    contrastive_loss3 = NtXentLoss(DEVICE)

    features3 = torch.ones((100, 128))
    transformed_features3 = torch.ones((10, 128))

    with pytest.raises(AssertionError):
        contrastive_loss3(features3, transformed_features3)
