from math import comb, exp, log, sqrt, pi
from typing import Optional, Tuple


class DDGAccountant:
    """Distributed Discrete Gaussian Mechanism with Privacy Amplification by Poisson Sub-sampling
    
    References 
        [DDG] The Distributed Discrete Gaussian Mechanism for Federated Learning with Secure Aggregation
              https://arxiv.org/pdf/2102.06387.pdf

        [PS] Poisson Subsampled Renyi Differential Privacy
             http://proceedings.mlr.press/v97/zhu19c/zhu19c.pdf
        
        [RDP] Renyi Differential Privacy
              https://arxiv.org/pdf/1702.07476.pdf
    
    Note
        1. The privacy calculations accounts for the combined effects of discrete Gaussian mechanism, 
        randomize rounding, model flattening, modular clipping, SGD clipping, see [DDG].

        2. All theoretical privacy in [DDG] are in terms of concentrated-DP. While the privacy effects 
        of sub-sampling accounted in [PS] are given in Renyi-DP. Therefore we shall convert accounting 
        to RDP throughout FL rounds using the result on Page 6 of [DDG]. Finally Renyi-DP shall be 
        converted to approximate-DP (namely epsilon-delta privacy). 
    """
    def __init__(
            self, 
            *,
            l2_clip: float,
            fl_rounds: int,
            noise_scale: float, 
            granularity: float, 
            model_dimension: int, 
            randomized_rounding_bias: float, 
            number_of_trustworth_fl_clients: int,
            poisson_subsampling_probability: float,
            approximate_dp_delta: Optional[float] = None,
            alpha_search_space_ends_inclusive: Tuple[int, int] = (2, 30)
    ) -> None:
        
        self.c = l2_clip
        self.d = model_dimension
        self.lb = alpha_search_space_ends_inclusive[0]
        self.ub = alpha_search_space_ends_inclusive[1]
        self.N = number_of_trustworth_fl_clients
        self.p = poisson_subsampling_probability
        self.rounds = fl_rounds
        self.beta = randomized_rounding_bias
        self.gamma = granularity
        self.sigma = noise_scale

        if approximate_dp_delta is None:
            # we want delta to be < 1/N where N is the client count
            self.delta = 0.9 / self.N
        else:
            self.delta = approximate_dp_delta

    def _delta_squared(self) -> float:
        """[DDG] Theorem 1, Equation 2"""

        c, d = self.c, self.d
        b, g = self.beta, self.gamma

        arg1 = c**2 + g**2 * d/4 + sqrt(2 * log(1/b)) * g * (c + g/2*sqrt(d))
        arg2 = (c + g * sqrt(d))**2

        return min(arg1, arg2)

    def _tau(self) -> float:
        """[DDG] Theorem 1, Equation 3"""

        common = -2 * (pi * self.sigma / self.gamma)**2

        tau = 0
        for k in range(1, self.N):
            tau += exp(common * k/(k+1))

        return 10 * tau

    def concentrated_dp_epsilon(self) -> float:
        """[DDG] Theorem 1, Equation 4"""

        common = self._delta_squared() / (self.N * self.sigma**2)
        d, t = self.d, self._tau()

        arg1 = sqrt(common + 2 * t * d)
        arg2 = sqrt(common) + t * sqrt(d)

        return min(arg1, arg2)
    
    def renyi_dp_epsilon(self, alpha: float) -> float:
        """Concentrated-DP to Renyi-DP. See [DDG] Page 6.
        
        Args 
            alpha: the RDP order 
        """

        assert alpha > 1

        eps = self.concentrated_dp_epsilon()

        return  alpha * eps**2 / 2
    
    def poisson_subsampled_rdp(self, alpha: int) -> float:
        """Privacy amplification when DDG is composed with Poisson sub-sampling. 
        
        Args
            alpha (int): In this accounting based on [PS] Theorem 5, we restrict alpha to
            be an integer at least 2 as alpha appears in a binomial coefficient in a finite sum.
        """

        assert isinstance(alpha, int) and alpha > 1

        p = self.p
        eps = self.renyi_dp_epsilon # this is a function

        arg1 = (1-p)**(alpha-1) * (alpha*p-p+1)
        arg2 = comb(alpha, 2) * p**2 * (1-p)**(alpha-2) * exp(eps(2))

        arg3, l = 0, 3
        while l <= alpha:
            arg3 += comb(alpha, l) * (1-p)**(alpha-l) * p**l * exp((l-1) * eps(l))
            l += 1
        arg3 *= 3

        return log(arg1+arg2+arg3) / (alpha-1)
    
    def fl_accountant_rdp(self, alpha: int, poisson_subsampled=False) -> float:
        """Returns rdp epsilon for a given rdp alpha resulting from composing 
        multiple federated rounds of [DDG] with Poisson subsampling."""
        if poisson_subsampled:
            return self.rounds * self.poisson_subsampled_rdp(alpha=alpha)
        return self.rounds * self.renyi_dp_epsilon(alpha=alpha)
    
    def approximate_dp_epsilon(self, alpha: int, poisson_subsampled=False) -> float:
        """See [RDP] Proposition 3."""
        rdp_eps = self.fl_accountant_rdp(alpha, poisson_subsampled)
        return rdp_eps + log(1/self.delta) / (alpha - 1)

    def optimal_adp_epslion(self, poisson_subsampled=False) -> float:
        """Get best approximate-DP epsilon."""
        optimal = float('inf')

        alpha = self.lb
        while alpha <= self.ub:
            candidate = self.approximate_dp_epsilon(alpha, poisson_subsampled)
            if candidate < optimal:
                optimal = candidate

            # NOTE print search process
            # print(f'alpha={alpha} candidate={candidate} optimal={optimal}')

            alpha += 1

        return optimal

if __name__ == '__main__':
    accountant = DDGAccountant(
            l2_clip = 3,
            noise_scale = 4, 
            granularity = 5, 
            model_dimension = 6, 
            randomized_rounding_bias = 0.7, 
            number_of_trustworth_fl_clients = 8,
            poisson_subsampling_probability=0.5,
            fl_rounds=10
    )

    print(accountant.optimal_adp_epslion())
