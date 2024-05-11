import torch
from math import sqrt, log
import matplotlib.pyplot as plt

def gaussian_mechanism(dim: int, stdev: float) -> torch.Tensor:
    return torch.normal(mean=0, std=stdev*torch.ones(dim))
