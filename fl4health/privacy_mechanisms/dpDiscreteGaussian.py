import math
from typing import List

import torch

# TODO rename file to discrete_gaussian.py, move tests to test/


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

    while True:
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
        List[int]: Privitized vector.
    """

    dim = len(query_vector)
    assert dim > 0  # query_vector needs to be nonempty

    return [query_vector[i] + discrete_gaussian_sampler(variance) for i in range(dim)]


if __name__ == "__main__":
    # # Test Cases for Discrete Gaussian
    # for _ in range(1,100):
    #     j = discrete_gaussian_sampler(1000 * math.pi)
    #     print(j)

    noised = discrete_gaussian_mechanism([*range(100)], 200)
    print(noised)

    pass
