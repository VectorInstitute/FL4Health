import torch

def gaussian_mechanism(dim: int, standard_deviation) -> torch.Tensor:
    stdev = standard_deviation*torch.ones(dim)
    return torch.normal(mean=0, std=stdev)


if __name__ == '__main__':
    print(gaussian_mechanism(10, 1))