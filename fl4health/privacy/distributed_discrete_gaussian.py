import math

import torch


class DiscreteGaussianPrivacy:
    """
    Privacy for sum of multidimensional discrete Gaussian.
    See mechanism and accounting found in https://arxiv.org/pdf/2102.06387.pdf
    """

    def __init__(self, sigma: float, alpha: float, discretized_vector: torch.Tensor, num_summands: int) -> None:

        # these checks come from Proposition 14 in https://arxiv.org/pdf/2102.06387.pdf
        assert sigma >= 0.5
        assert alpha >= 1

        self.sigma = sigma  # standard deviation
        self.var = sigma**2
        self.alpha = alpha  # order of Rényi divergence
        self.l1_norm = torch.linalg.vector_norm(discretized_vector, ord=1)
        self.l2_norm_squared = torch.dot(discretized_vector, discretized_vector)
        self.l2_norm = torch.sqrt(self.l2_norm_squared)
        self.dim = discretized_vector.size().numel()
        self.N = num_summands
        self.tau = self.calculate_tau()

    def calculate_tau(self) -> float:
        tau: float = 0
        for k in range(1, self.N):
            tau += math.exp(-2 * (math.pi * self.sigma) ** 2 * k / (k + 1))
        tau *= 10
        return tau

    def rdp_epsilon(self) -> float:

        # denominators
        d = math.sqrt(self.N) * self.sigma
        d_squared = self.N * self.var

        half = self.alpha / 2

        l1 = self.l1_norm.item()
        l2 = self.l2_norm.item()
        l2_squared = self.l2_norm_squared.item()

        # bounds
        b1 = half * l2_squared / d_squared + self.tau * self.dim
        b2 = half * (l2_squared / d_squared + 2 * l1 * self.tau / d + self.tau**2 * self.dim)
        b3 = half * (l2 / d + self.tau * math.sqrt(d)) ** 2

        return min([b1, b2, b3])


# TODO create tests
