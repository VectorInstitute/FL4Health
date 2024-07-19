import torch

from fl4health.utils.functions import bernoulli_sample, sigmoid_inverse


def test_bernoulli_gradient() -> None:
    torch.manual_seed(42)
    theta = torch.rand(7)
    theta.requires_grad = True
    pred = bernoulli_sample(theta)
    target = torch.ones(7)
    loss = torch.sum((pred - target) ** 2)
    loss.backward()
    assert (theta.grad == 2 * (pred - target) * theta).all()
    torch.seed()


def test_sigmoid_inverse() -> None:
    torch.manual_seed(42)
    x = torch.rand(7)
    z = torch.sigmoid(x)
    assert torch.allclose(sigmoid_inverse(z), x)
    torch.seed()
