import torch
from math import sqrt, log

def gaussian_mechanism(dim: int, epsilon: float, delta: float, clip: float) -> torch.Tensor:
    """
    Ref https://proceedings.mlr.press/v80/balle18a/balle18a.pdf
    """
    standard_deviation = clip * sqrt(2*log(1.25/delta)) / epsilon
    stdev = standard_deviation*torch.ones(dim)
    return torch.normal(mean=0, std=stdev)


if __name__ == '__main__':
    print(gaussian_mechanism(10, 1, 0.01, 5))