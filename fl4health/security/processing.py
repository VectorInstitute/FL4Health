"""Pre-processing and post processing utils
"""
from typing import List

import torch
import torch.linalg as LA
from scipy.linalg import hadamard


def generate_diagonal_matrix(d: int) -> torch.Tensor:
    """Generate d dimensional diagonal matrix
    whose diagonal entries are -1 and 1 with equal probability.
    """
    p = 0.5

    # generate once, use twice
    ones = torch.ones(d)

    p_mat = p * ones
    diag = 2 * torch.bernoulli(p_mat) - ones

    return torch.diag(diag)


def l2_norm(model: torch.nn.Module) -> torch.tensor:
    sum = torch.zeros(1)
    for key, val in model.state_dict().items():
        # take component wise products
        sum += torch.sum(val * val)
    return torch.sqrt(sum)


def vectorize_model(model: torch.nn.Module) -> torch.tensor:
    layers = []
    for _, layer in model.state_dict().items():
        layers.append(torch.flatten(layer))
    return torch.cat(layers)


def padded_vectorized_model(model: torch.nn.Module) -> torch.tensor:
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    vec = vectorize_model(model)
    d = len(vec)
    assert d == num_trainable_params

    exponent = torch.log2(d)
    padding_size = 1 << torch.ceil(exponent) - d

    return torch.cat([vec, torch.zeros(padding_size)])


def clip(model: torch.nn.Module, c, gamma) -> torch.tensor:
    norm = l2_norm(model)
    v = padded_vectorized_model(model)
    scalar = min(1, c / norm) / gamma
    return scalar * v


def get_model_signature(model: torch.nn.Module) -> List[torch.Size]:
    signatures = []
    for _, layer in model.state_dict().items():
        signatures.append(layer.size())
    return torch.cat(signatures)


def compress(model: torch.nn.Module) -> List[torch.Size]:
    # review required
    vec = padded_vectorized_model(model)
    d = len(vec)

    H = hadamard(d)
    D = generate_diagonal_matrix(d)
    return torch.matmul(H, torch.matmul(D, vec))


def devectorize_model(vect, size_array) -> torch.nn.Module:
    # to review
    vect_array = []
    i = 0
    for size in size_array:
        j = i + size.item()
        unflattend = torch.unflatten(vect[i:j], 1, size)
        vect_array.append(unflattend)
        i = j
    return vect_array
