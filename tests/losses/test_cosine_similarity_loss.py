import pytest
import torch

from fl4health.losses.cosine_similarity_loss import CosineSimilarityLoss


DEVICE = torch.device("cpu")


def test_cosine_similarity_loss() -> None:
    cosine_similarity_loss_function = CosineSimilarityLoss(DEVICE)
    # Similar vectors should be 1, opposite should be 1 (because we take absolute value), orthogonal should be zeros
    # mean should be 2/3
    first_batch = torch.Tensor([[1, 2, 3], [1, 2, 3], [1, 0, 1]])
    second_batch = torch.Tensor([[1, 2, 3], [-1, -2, -3], [0, 2, 0]])
    loss = cosine_similarity_loss_function(first_batch, second_batch).item()
    assert pytest.approx(loss, abs=0.0001) == 2.0 / 3.0


def test_cosine_similarity_dim_mismatch() -> None:
    cosine_similarity_loss_function = CosineSimilarityLoss(DEVICE)
    # Similar vectors should be 1, opposite should be 1 (because we take absolute value), orthogonal should be zeros
    # mean should be 2/3
    first_batch = torch.Tensor([[1, 2, 3], [1, 2, 3], [1, 0, 1]])
    second_batch = torch.Tensor([[1, 2, 3], [-1, -2, -3], [0, 2, 0], [1, 2, 3]])
    with pytest.raises(AssertionError) as assertion_exception:
        cosine_similarity_loss_function(first_batch, second_batch)

    assert str(assertion_exception.value) == "Tensors have different batch sizes"
