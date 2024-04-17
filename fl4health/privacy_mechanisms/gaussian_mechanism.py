import torch
from math import sqrt, log
import matplotlib.pyplot as plt

def gaussian_mechanism(dim: int, stdev: float) -> torch.Tensor:
    return torch.normal(mean=0, std=stdev*torch.ones(dim))

# def old_gaussian_mechanism(dim: int, epsilon: float, delta: float, clip: float) -> torch.Tensor:
#     """
#     Ref https://proceedings.mlr.press/v80/balle18a/balle18a.pdf
#     """
#     standard_deviation = clip * sqrt(2*log(1.25/delta)) / epsilon
#     stdev = standard_deviation*torch.ones(dim)
#     stdev=torch.ones(25)
#     return torch.normal(mean=0, std=stdev)


if __name__ == '__main__':
    print(gaussian_mechanism(dim=10, stdev=0.0001))