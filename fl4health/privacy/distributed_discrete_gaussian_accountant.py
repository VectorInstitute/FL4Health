from math import comb, exp, log, sqrt, pi, e
from typing import Optional, Tuple
from functools import partial

def rdp_subsampling_amplification(subsampling_ratio, alpha, rdp_epsilon_func):
    """
    Reference
        Theorem 9 in "Subsampled Rényi Differential Privacy and Analytical Moments Accountant"
        https://www.shivakasiviswanathan.com/AISTATS19.pdf
    """
    assert alpha > 1 and isinstance(alpha, int)

    # alias
    g = subsampling_ratio
    eps = rdp_epsilon_func
    e2 = exp(eps(2))

    # summands
    a = g**2 * comb(alpha, 2) * min(4*(e2-1), 2 * e2)
    b = sum(g**j * comb(alpha, j)*exp((j-1)*eps(j)) * 2 for j in range(3, alpha+1))
    
    return log(1+a+b) / (alpha-1)


class DistributedDiscreteGaussianAccountant:
    """Distributed Discrete Gaussian Mechanism with Privacy Amplification by Poisson Sub-sampling
    
    References 
        [DDG] The Distributed Discrete Gaussian Mechanism for Federated Learning with Secure Aggregation
              https://arxiv.org/pdf/2102.06387.pdf

        [DDP-J] Journal version of [DDG]
                http://proceedings.mlr.press/v139/kairouz21a/kairouz21a.pdf

        [PS] Poisson Subsampled Renyi Differential Privacy
             http://proceedings.mlr.press/v97/zhu19c/zhu19c.pdf
        
        [RDP] Renyi Differential Privacy
              https://arxiv.org/pdf/1702.07476.pdf
        
        [zCDP] Concentrated Differential Privacy
            https://arxiv.org/pdf/1605.02065.pdf
    
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
            alpha_search_space = (1, 10),
            randomized_rounding_bias: float, 
            number_of_trustworthy_fl_clients: int,
            approximate_dp_delta: Optional[float] = None,
            privacy_amplification_sampling_ratio: float = 1,
            poisson_subsampling_probability: Optional[float] = None
    ) -> None:
        
        self.c = l2_clip
        self.d = model_dimension
        self.lb = alpha_search_space[0]
        self.ub = alpha_search_space[1]
        self.g = privacy_amplification_sampling_ratio
        self.N = number_of_trustworthy_fl_clients
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

        arg1 = c**2 + g**2 * d/4 + sqrt(2 * log(1/b)) * g * (c + g*sqrt(d)/2)
        arg2 = (c + g * sqrt(d))**2

        return min(arg1, arg2)

    def _tau(self) -> float:
        """[DDG] Theorem 1, Equation 3"""

        common = -2 * (pi * self.sigma / self.gamma)**2

        tau = 0
        for k in range(1, self.N):
            tau += exp(common * k/(k+1))

        return 10 * tau

    def concentrated_dp_epsilon_single_round(self) -> float:
        """[DDG] Theorem 1, Equation 4"""

        common = self._delta_squared() / (self.N * self.sigma**2)
        d, t = self.d, self._tau()

        arg1 = sqrt(common + 2 * t * d)
        arg2 = sqrt(common) + t * sqrt(d)

        return min(arg1, arg2)
    
    def renyi_dp_single_round(self, alpha: float):
        """
        We use the fact that eps/2-concentrated DP is the same as 
        (alpha, eps^2 * alpha/2)-RDP for all alpha > 1.

        Reference
            Page 2 of [DDG-J]
        """
        assert alpha > 1
        cdp_eps = self.concentrated_dp_epsilon_single_round() 
        return .5 * cdp_eps**2 * alpha


    def fl_rdp_accountant(self, alpha, amplify=True):
        if amplify:
            return self.rounds * rdp_subsampling_amplification(subsampling_ratio=self.g, alpha=alpha, rdp_epsilon_func=self.renyi_dp_single_round)
        return self.rounds * self.renyi_dp_single_round(alpha)
    
    def fl_approximate_dp_accountant(self, amplify=False):
        # search scale is only used when amplify=False
        search_scale=0.0001

        # alpha_integer_upper_bound is only used when amplify=True
        alpha_integer_upper_bound = 3

        if not amplify:
            # convert concentrated to approximate-dp according to Lemma 4 in [DDG-J] 
            concentrated_eps = self.rounds * self.concentrated_dp_epsilon_single_round()

            def compute_candidate(alpha: float, concentrated_eps: float):
                assert alpha > 1
                u = concentrated_eps**2 * alpha/2
                v = log(  1 / (alpha*self.delta)  ) / (alpha-1)
                w = log(1-1/alpha)
                return u + v + w
        
            optimial_value = float('inf')
                
            alpha = self.lb + search_scale
            while alpha <= self.ub:
                candidate_value = compute_candidate(alpha=alpha, concentrated_eps=concentrated_eps)
                if candidate_value < optimial_value:
                    optimial_value = candidate_value
                    # print(alpha, optimial_value)
                alpha += search_scale
            
            return optimial_value
        
        else:
            # convert Renyi-DP to approximate-DP by [RDP] Proposition 3
            optimial_value = float('inf')
            for alpha in range(2, alpha_integer_upper_bound):
                rdp_eps = self.fl_rdp_accountant(alpha=alpha, amplify=True)
                candidate_value = rdp_eps + log(1/self.delta) / (alpha-1)
                if candidate_value < optimial_value:
                    optimial_value = candidate_value
                    print('amplified: ', alpha, optimial_value)
            return optimial_value




if __name__ == '__main__':
    privacy_parameters = {
        "clipping_threshold": 0.001,
        "noise_scale": 0.0004, #0.005,
        "granularity": 0.000_000_1, #0.001,
        "bias": exp(-0.5),
        "dp_mechanism": "discrete gaussian mechanism",
        "model_dimension": 500_000, #1_148_066,
        "num_fl_rounds": 1,
        "num_clients": 4,
        'privacy_amplification_sampling_ratio': 0.1,
        'approximate_dp_delta': 1/23247**2,
    }

    accountant = DistributedDiscreteGaussianAccountant(
        l2_clip=privacy_parameters["clipping_threshold"],
        noise_scale=privacy_parameters["noise_scale"],
        granularity=privacy_parameters["granularity"],
        model_dimension=privacy_parameters["model_dimension"],
        randomized_rounding_bias=privacy_parameters["bias"],
        number_of_trustworthy_fl_clients=privacy_parameters["num_clients"],
        fl_rounds=privacy_parameters["num_fl_rounds"],
        privacy_amplification_sampling_ratio=privacy_parameters['privacy_amplification_sampling_ratio'],
        approximate_dp_delta=privacy_parameters['approximate_dp_delta']
    )

    # coarse_clips = [1, 0.1, 0.01, 0.001, 0.0001, 0.00001]
    # fine_clips1 = [0.001, 0.003, 0.005, 0.007, 0.009, 0.01]
    # fine_clips2 = [0.02, 0.03, 0.05, 0.07, 0.09, 0.1]
    # # best clip = 0.01

    # # 0.00001 0.0001 0.0003 0.0005 0.0007 0.001
    # granularities = [0.00001, 0.0001, 0.0003, 0.0005, 0.0007, 0.001]
    # # granularities = [0.0001]
    # # 0.0001 0.0003 0.0005 0.0007 0.0009
    # noises = [0.0001, 0.0003, 0.0005, 0.0007, 0.0009]
    # # best 0.0009
    # epsilons = []
    # for x in granularities:
    #     accountant.gamma = x
    #     # eps_rdp = accountant.fl_rdp_accountant(alpha=2)
    #     eps_approx = accountant.fl_approximate_dp_accountant(amplify=False) # 
    #     epsilons.append(eps_approx)
    #     # print('hyperparm, approx-dp, rdp: ')
    #     print(x, eps_approx)
    #     # print('tau and delta squared: ', accountant._tau(), accountant._delta_squared())
    # print(epsilons)

    print(accountant.fl_approximate_dp_accountant(amplify=False))