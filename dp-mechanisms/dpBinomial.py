import torch 
from typing import List

# TODO why doesn't mechanism result in output type List[int]
def BinomialMechanism(query_vector: List[int], N: int, p: float, j: int) -> List[int]:
    """Privatization of query_vector via additive binomial noise.

    Args 
        query_vector
            A vector with integer valued components 
        N, p
            The number N of Bernoulli trials with success probability p which determines the distribution Bin(N, p).
        j
            Determines the quantization scale s = 1/j

    Return 
        Privatized query vector via the scaler binomial mechanism applied to each component of query_vector:

    Reference 
        "cpSGD: Communication-efficient and differentially-private distributed SGD"
        https://proceedings.neurips.cc/paper_files/paper/2018/file/21ce689121e39821d07d04faab328370-Paper.pdf
    """
    dim = len(query_vector)
    assert dim > 0      # nonempty query_vector 

    binomial_samples = torch.distributions.Binomial(dim, p * torch.ones(dim)).sample()
    query = torch.tensor(query_vector)

    privatized_vector = query + (binomial_samples - N * p) / j

    return privatized_vector.tolist()


if __name__ == '__main__':
    out = BinomialMechanism([*range(10)], 20, 0.5, 6)
    print(out, type(out))
