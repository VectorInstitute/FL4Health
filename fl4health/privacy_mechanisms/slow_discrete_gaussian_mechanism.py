import math
from logging import WARNING, INFO, DEBUG
from typing import List

import numpy as np
import torch
from flwr.common.logger import log
from torch.linalg import vector_norm



def bernoulli_exp(gamma: float) -> int:
    """
    Draws a sample from Bernoulli(exp(-gamma))

    Reference:
        Algorithm 2 of "The Discrete Gaussian for Differential Privacy"
        https://proceedings.neurips.cc/paper/2020/file/b53b3a3d6ab90ce0268229151c9bde11-Paper.pdf

    Args:
        gamma (float)

    Returns:
        int: A Bernoulli sample, either 0 or 1

    """

    assert gamma >= 0  # ensures probabiity is not over 1

    if gamma <= 1:
        K, g = 1, torch.tensor(float(gamma))
        sample_A = torch.bernoulli(g)

        while sample_A == 1:
            K += 1
            sample_A = torch.bernoulli(g / K)

        return K % 2

    # if gamma > 1
    G, p = math.floor(gamma), torch.exp(torch.tensor(-1))

    for _ in range(G):
        sample_B = bernoulli_exp(p.item())
        if sample_B == 0:
            return 0

    delta = G - gamma  # delta is in [0, 1]

    p = torch.exp(torch.tensor(delta))

    return bernoulli_exp(p.item())  # sample_C


def discrete_gaussian_sampler(variance: float) -> int:
    """
    Takes a sample from the centered discrete Gaussian distribution with given variance, using rejection sampling.

    Reference
        Algorithm 1 in "The Discrete Gaussian for Differential Privacy"
        https://proceedings.neurips.cc/paper/2020/file/b53b3a3d6ab90ce0268229151c9bde11-Paper.pdf

    Args:
        variance (float): Variance.

    Returns:
        int: An integer sample from the centered discrete Gaussian distribution with given variance.
    """

    assert variance > 0  # Requires variance > 0

    t = math.floor(math.sqrt(variance)) + 1

    counter = 0
    while True:
        counter += 1
        # log(DEBUG, f'discrete_gaussian_sampler round {counter}')
        # if counter % 100 == 0:
            # log(WARNING, f"Discrete Gaussian Sampling has completed {counter} samples without success.")
        U = torch.randint(0, t, (1,)).item()
        D = bernoulli_exp(U / t)

        if D == 0:
            continue  # reject

        # generate V from Geometric(1 − 1/e)
        V = 0
        while True:
            sample_A = bernoulli_exp(1)
            if sample_A == 0:
                break
            V += 1

        sample_B = torch.bernoulli(torch.tensor(0.5))

        if sample_B == 1 and U == 0 and V == 0:
            continue  # reject

        Z = (1 - 2 * sample_B) * (U + t * V)  # discrete Laplace

        gamma = (torch.abs(Z) - variance / t) ** 2 / (2 * variance)

        sample_C = bernoulli_exp(gamma.item())

        if sample_C == 0:
            continue  # reject

        # EXIT LOOP
        return int(Z.item())


def discrete_gaussian_mechanism(query_vector: List[int], variance: float) -> List[int]:
    """
    Applies additive discrete Gaussian noise to query_vector.

    Reference
        The Distributed Discrete Gaussian Mechanism for Federated Learning with Secure Aggregation
        http://proceedings.mlr.press/v139/kairouz21a/kairouz21a.pdf

    Args:
        query_vector (List[int]): This is the discretized gradient vector in the SecAgg context.
        variance (float): Gaussian noise variance.

    Returns:
        List[int]: privatized vector.
    """

    assert len(query_vector) > 0  # query_vector needs to be nonempty

    return [query_element + discrete_gaussian_sampler(variance) for query_element in query_vector]


def discrete_gaussian_noise_vector(d: int, variance: float) -> torch.Tensor:
    """d-dimensional centered discrete Gaussian random vector with independent components of given variance."""
    # TODO paralleize this loop more efficiently
    # sampler = torch.vmap(lambda variance: torch.where(True, discrete_gaussian_sampler(variance), 0))
    # return sampler(d * torch.ones(d))
    vectr = []
    for _ in range(d):
        vectr.append(discrete_gaussian_sampler(variance))
    return torch.tensor(vectr)


def random_sign_vector(dim: int, sampling_probability: float) -> torch.Tensor:
    """A random tensor of +/- ones used on the client-side as part of the discrete Gaussian mechanism."""
    return 2 * torch.bernoulli(sampling_probability * torch.ones(dim)) - torch.ones(dim)


def generate_sign_diagonal_matrix(dim: int, sampling_probability=0.5, seed=0) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.diag(random_sign_vector(dim=dim, sampling_probability=0.5))

def generate_random_sign_vector(dim: int, sampling_probability=0.5, seed=0) -> torch.Tensor:
    torch.manual_seed(seed)
    return random_sign_vector(dim=dim, sampling_probability=0.5)


def pad_zeros(vector: torch.Tensor, dim=None) -> torch.Tensor:
    """Elongate vector dimension to next power of two."""
    assert vector.dim() == 1
    if dim is None:
        dim = vector.numel()
    exp = get_exponent(dim)
    pad_len = torch.pow(torch.tensor(2), exp) - dim
    pad_len = pad_len.to(int).item()
    return torch.cat((vector, torch.zeros(pad_len)))

def get_exponent(n: int) -> int:
    """Get exponent of the least power of two greater than or equal to n."""
    return math.ceil(math.log2(n))

def generate_walsh_hadamard_matrix(exponent: int) -> torch.Tensor:
    """The dimension of the matrix is 2^exponent. This matrix is its inverse."""

    assert isinstance(exponent, int) and exponent >= 0

    def sylvester(exponent: int) -> np.ndarray:
        if exponent == 0:
            return np.ones(1, dtype=int)

        block = sylvester(exponent - 1)

        return np.block([[block, block], [block, -block]])

    return torch.from_numpy(sylvester(exponent) / np.sqrt(2**exponent))


def randomized_rounding(vector: torch.Tensor, clip: float, granularity: float, unpadded_model_dim: int, bias: float) -> torch.Tensor:
    """Random rounding for SecAgg client procedure.

    Ref http://proceedings.mlr.press/v139/kairouz21a/kairouz21a.pdf
    """
    assert 0 <= bias < 1
    device = vector.device
    log(INFO, f'randomized rounding using device: {device}')

    r = clip / granularity
    padded_dim = 2**get_exponent(unpadded_model_dim)
    s = math.sqrt(padded_dim)

    upper_bound_1 = r + s
    upper_bound_2 = r**2 + padded_dim/4 + math.sqrt(2 * math.log(1/bias)) * (r + s/2)
    upper_bound_2 = math.sqrt(upper_bound_2)

    upper_bound = min(upper_bound_1, upper_bound_2)

    #     # TODO paralleize this for loop with Numba
    # def round(vector, round_down_probabilities, dim) -> torch.Tensor:
    #     rounding_instructions = torch.bernoulli(round_down_probabilities)

    #     for i in range(dim):
    #         if rounding_instructions[i] == 1:
    #             # if true round down
    #             vector[i] = torch.floor(vector[i])
    #             continue
    #         vector[i] = torch.ceil(vector[i])
    #     return vector

    rounder = torch.vmap(lambda x, coin: torch.where(coin==1, torch.floor(x), torch.ceil(x)))

    down_probs = torch.ceil(vector) - vector
    purse = torch.bernoulli(down_probs)

    rounded_vector = rounder(vector, purse)
    i = 0
    while torch.linalg.vector_norm(rounded_vector, ord=2) > upper_bound:
        n  = torch.linalg.vector_norm(rounded_vector, ord=2)
        log(INFO, f'round: {i} {n}>{upper_bound}')
        i+=1
        purse = torch.bernoulli(down_probs)
        rounded_vector = rounder(vector, purse)
    log(DEBUG, f'Done rounding')
    return rounded_vector


def clip_vector(vector: torch.Tensor, granularity: float, clip: float) -> torch.Tensor:
    assert vector.dim() == 1  
    if vector.dtype is not torch.float64:
        vector = vector.to(dtype=torch.float64)
    scalar = min(1, clip/vector_norm(vector)) / granularity
    return scalar * vector

if __name__ == '__main__':
    # v = torch.tensor([3,4])
    # clipped = clip_vector(v, 10, 0.5)
    # print(clipped)
    torch.set_default_device('cuda')
    # for i in range(1, 17):
    #     print(i, get_exponent(i))
    v = torch.rand(8)
    # down_probs = torch.ceil(v) - v

    # rounder = torch.vmap(lambda x, coin: torch.where(coin==1, torch.floor(x), torch.ceil(x)))
    # y = rounder(v, down_probs)
    # print(v, y)
    # w = randomized_rounding(v, 10, 0.4, 8, 0.5)
    # print(w)


