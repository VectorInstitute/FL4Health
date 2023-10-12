import math
from typing import List

import torch


def Bernoulli_exp(gamma: float) -> int:
    """Draws a sample from Bernoulli(exp(-gamma))

    Args
        gamma
            A non-negative parameter of type float

    Return
        A Bernoulli sample, either 0 or 1

    Raises
        AssertionError

    Reference
        Algorithm 2 in "The Discrete Gaussian for Differential Privacy"
        https://proceedings.neurips.cc/paper/2020/file/b53b3a3d6ab90ce0268229151c9bde11-Paper.pdf
    """

    assert gamma >= 0  # ensures probabiity is not over 1

    if 0 <= gamma <= 1:
        K, g = 1, torch.tensor(float(gamma))
        sample_A = torch.bernoulli(g)

        while sample_A == 1:
            K += 1
            sample_A = torch.bernoulli(g / K)

        return K % 2

    # if gamma > 1
    G, p = math.floor(gamma), torch.exp(torch.tensor(-1))

    for _ in range(G):
        sample_B = torch.bernoulli(p)
        if sample_B == 0:
            return 0

    delta = G - gamma  # delta is in [0, 1]

    p = torch.exp(torch.tensor(delta))

    return Bernoulli_exp(p.item())  # sample_C


def DiscreteGaussianSampler(variance: float) -> int:
    """Takes a sample from the centered discrete Gaussian distribution with given variance, using rejection sampling.

    Args
        variance
            Positive float

    Return
        An integer sample from the centered discrete Gaussian distribution with given variance.

    Raises
        AssertionError

    Reference
        Algorithm 1 in "The Discrete Gaussian for Differential Privacy"
        https://proceedings.neurips.cc/paper/2020/file/b53b3a3d6ab90ce0268229151c9bde11-Paper.pdf
    """

    assert variance > 0  # Requires variance > 0

    t = 1 + math.floor(variance)

    while True:
        U = torch.randint(0, t, (1,)).item()
        D = Bernoulli_exp(U / t)

        if D == 0:
            continue  # reject

        # generate V from Geometric(1 − 1/e)
        V = 0
        while True:
            sample_A = Bernoulli_exp(1)
            if sample_A == 0:
                break
            V += 1

        sample_B = torch.bernoulli(torch.tensor(0.5))

        if sample_B == 1 and U == 0 and V == 0:
            continue  # reject

        Z = (1 - 2 * sample_B) * (U + t * V)  # discrete Laplace

        gamma = (torch.abs(Z) - variance / t) ** 2 / (2 * variance)

        sample_C = Bernoulli_exp(gamma.item())

        if sample_C == 0:
            continue  # reject

        # EXIT LOOP
        return int(Z.item())


def DiscreteGaussianMechanism(query_vector: List[int], variance: float) -> List[int]:
    """Applies additive discrete Gaussian noise to query_vector.

    Args
        query_vector
            This is the discretized gradient vector in the SecAgg context.
        variance
            For the Gaussian noise

    Return
        Privitized vector of type List[int]

    Raises
        AssertionError

    Reference
        "The Distributed Discrete Gaussian Mechanism for Federated Learning with Secure Aggregation"
        http://proceedings.mlr.press/v139/kairouz21a/kairouz21a.pdf
    """

    dim = len(query_vector)
    assert dim > 0  # query_vector needs to be nonempty

    return [query_vector[i] + DiscreteGaussianSampler(variance) for i in range(dim)]


if __name__ == "__main__":
    # # Test Cases for Discrete Gaussian
    # for _ in range(1,100):
    #     j = DiscreteGaussianSampler(1000 * math.pi)
    #     print(j)

    noised = DiscreteGaussianMechanism([*range(100)], 200)
    print(noised)

    pass
