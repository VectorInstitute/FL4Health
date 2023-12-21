from typing import List

import torch

# TODO rename file to binomial.py, move tests to test/


def binomial_mechanism(query_vector: List[int], N: int, p: float, j: int) -> List[float]:
    """
    Privatization of query_vector via additive binomial noise.

    Reference
        cpSGD: Communication-efficient and differentially-private distributed SGD
        https://proceedings.neurips.cc/paper_files/paper/2018/file/21ce689121e39821d07d04faab328370-Paper.pdf

    Args:
        query_vector (List[int]): A vector with integer valued components.
        N (int): The number N of Bernoulli trials which determines the distribution Bin(N, p).
        p (float): The success probability p which determines the distribution Bin(N, p).
        j (int): Determines the quantization scale s = 1/j.

    Returns:
        List[int]: Privatized query vector via the scaler binomial mechanism applied to each component of query_vector
    """

    dim = len(query_vector)
    assert dim > 0  # nonempty query_vector

    binomial_samples = torch.distributions.Binomial(dim, p * torch.ones(dim)).sample()
    query = torch.tensor(query_vector)

    privatized_vector = query + (binomial_samples - N * p) / j

    return privatized_vector.tolist()


if __name__ == "__main__":
    out = binomial_mechanism([*range(10)], 20, 0.5, 6)
    print(out, type(out))
