import pytest
import torch

from fl4health.losses.contrastive_loss import ContrastiveLoss

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def test_contrastive_loss() -> None:
    # Default temperature is 0.5
    contrastive_loss = ContrastiveLoss(DEVICE)

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
    contrastive_loss = ContrastiveLoss(DEVICE)
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
