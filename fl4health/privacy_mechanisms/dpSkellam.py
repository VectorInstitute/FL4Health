import torch


def SkellamMechanism(query_vector: list[int], skellam_variance: float) -> list[int]:
    """Additive mechanism that adds to the given query vector a discrete noise vector
    sampled from the centered Skellam distribution of given variance.

    Args
        query_vector
            Discritized vector, meaning its type is an integer list, to which noise will be added.
        skellam_variance
            Variance associated with each component of the d-dimensional Skellam distribution.

    Return
        Perturbed vector of type [int]

    Notes
        1) A Skellam random variable can be obtained as the difference of two independent Poisson random variables.
        2) A Skellam random vector has as components Skellam random variables.
        3) Computations are carried throughout on torch.int (32-bit signed integer)

    Reference
        "The Skellam Mechanism for Differentially Private Federated Learning"
        https://proceedings.neurips.cc/paper/2021/file/285baacbdf8fda1de94b19282acd23e2-Paper.pdf
    """

    dim = len(query_vector)

    if dim == 0:
        raise Exception("query_vector cannot be empty")

    if type(query_vector[0]) != int:
        raise Exception("Did you forget to discretize the gradient vector? query_vector must be a list of integers")

    # mean of Possion variables used to construct a
    # centered Skellam variable S with Var(S) = skellam_variance
    mean = skellam_variance / 2
    mean_vector = torch.ones(dim) * mean

    Skellam = torch.poisson(mean_vector) - torch.poisson(mean_vector)
    Skellam = Skellam.int()  # convert to 32-bit signed integer

    perturbed = torch.tensor(query_vector, dtype=torch.int) + Skellam  # add noise

    return perturbed.tolist()


if __name__ == "__main__":
    # test cases & type checking
    var = 40
    vec = [*range(-10, 10)]
    out = SkellamMechanism(vec, var)

    print(out, type(out), type(out[0]))
