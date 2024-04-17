"""References 

    [Amplify] Wang et al, "Subsampled Rényi Differential Privacy and Analytical Moments Accountant"
        https://www.shivakasiviswanathan.com/AISTATS19.pdf

    [DDG] Kairouz et al, "The Distributed Discrete Gaussian Mechanism for Federated Learning with Secure Aggregation"
            https://arxiv.org/pdf/2102.06387.pdf (This arXiv version contains detailed proofs.)

    [DDP-J] Journal version of [DDG]
            http://proceedings.mlr.press/v139/kairouz21a/kairouz21a.pdf

    [RDP] Mironov, "Renyi Differential Privacy"
        https://arxiv.org/abs/1702.07476

"""

from typing import Any, Callable, Optional
from math import comb, exp, log, pi, sqrt

def gaussian_rdp(rdp_order: float, standard_deviation: float) -> float:
    assert rdp_order > 1

    # See [RDP] Table II
    rdp_epsilon = rdp_order / (2 * standard_deviation**2)

    return rdp_epsilon

def subsampled_accountant_factory(rdp_epsilon_function: Callable[[float], float], sampling_ratio: float) -> Callable[[int], float]:
    """Returns the RDP accountant with privacy amplification. 
    
    Important assumption:
        rdp_epsilon_function(alpha) -> inf as alpha -> inf
    """

    assert 0 < sampling_ratio <= 1

    # Alias
    eps = rdp_epsilon_function
    ee2 = exp(eps(2))

    def subsampled_mechanism_rdp(rdp_order: int) -> float:
        assert rdp_order > 1 and isinstance(rdp_order, int)

        # See [Amplify] Theorem 9

        a = sampling_ratio**2 * comb(rdp_order, 2) * min(4*(ee2-1), 2*ee2)
        b = sum(sampling_ratio**j * comb(rdp_order, j) * exp((j-1)*eps(j)) * 2 for j in range(3, rdp_order+1))


        rdp_epsilon = log(1+a+b) / (rdp_order-1)

        return rdp_epsilon
    
    return subsampled_mechanism_rdp

def is_positive(*numbers: Any) -> None:
    for n in numbers:
        assert n > 0

def is_natural(*numbers: Any) -> None:
    for n in numbers:
        assert n > 0 and isinstance(n, int)

class DDGaussAccountant:
    def __init__(self, 
                l2_norm_clip: float,
                private_vector_dimension: int,
                granularity: float, 
                noise_scale: float, 
                n_trustworthy_clients: int,
                n_fl_rounds: int,
                privacy_amplification_sampling_ratio: float,
                bias: float = exp(-0.5), 
                approximate_dp_delta: Optional[float] = None
    ) -> None:
        
        is_natural(private_vector_dimension, n_trustworthy_clients, n_fl_rounds)
        is_positive(l2_norm_clip, granularity, noise_scale)
        assert 0 < privacy_amplification_sampling_ratio <= 1
        assert 0 <= bias <= 1
        assert approximate_dp_delta >=0 and isinstance(approximate_dp_delta, float)
        
        # Alias
        self.beta = bias
        self.c = l2_norm_clip
        self.d = private_vector_dimension
        self.gamma = granularity
        self.N = n_trustworthy_clients
        self.p = privacy_amplification_sampling_ratio
        self.rounds = n_fl_rounds
        self.sigma = noise_scale

        if approximate_dp_delta is None:
            self.delta = 1 / n_trustworthy_clients**2
        else:
            self.delta = approximate_dp_delta

    # this delta is different from the approximate dp delta
    def _delta_squared(self) -> float:
        """[DDG-J] Theorem 5"""

        c, d = self.c, self.d
        b, g = self.beta, self.gamma

        arg1 = c**2 + g**2 * d/4 + sqrt(2 * log(1/b)) * g * (c + g*sqrt(d)/2)
        arg2 = (c + g * sqrt(d))**2

        return min(arg1, arg2)

    def _tau(self) -> float:
        """[DDG-J] Theorem 5"""

        common = -2 * (pi * self.sigma / self.gamma)**2

        tau = 0
        for k in range(1, self.N):
            tau += exp(common * k/(k+1))

        return 10 * tau

    def single_fl_round_concentrated_dp(self) -> float:
        """[DDG-J] Theorem 5"""

        common = self._delta_squared() / (self.N * self.sigma**2)
        d, t = self.d, self._tau()

        arg1 = sqrt(common + 2 * t * d)
        arg2 = sqrt(common) + t * sqrt(d)

        epsilon = min(arg1, arg2)

        # 1 FL round satisfies (epsilon^2)/2 - concentrated differential privacy
        return epsilon
    
    def single_fl_round_renyi_dp(self, rdp_order: float) -> float:

        assert rdp_order > 1

        # See [DDG-J] comment above Lemma 4
        cdp_epsilon = self.single_fl_round_concentrated_dp()
        rdp_epsilon = rdp_order/2 * cdp_epsilon **2 

        # 1 FL round satisfies (rdp_order, rdp_epsilon) - Renyi differential privacy
        return rdp_epsilon
    
    def fl_rdp(self, rdp_order: float|int, amplify=False) -> float:

        assert rdp_order > 1 
        rdp_accountant = self.single_fl_round_renyi_dp
        rdp_epsilon = rdp_accountant(rdp_order)

        if amplify:
            # Privacy amplification only accepts integer rdp order, see [Amplify] Theorem 9
            assert isinstance(rdp_order, int)
            amplified_rdp_accountant = subsampled_accountant_factory(rdp_epsilon_function=rdp_accountant, sampling_ratio=self.p)

            # It is possible the amplified epsilon > unamplified epsilon 
            rdp_epsilon = min(rdp_epsilon, amplified_rdp_accountant(rdp_order))

        return self.rounds * rdp_epsilon

    def fl_approximate_dp(self, amplify=False, verbose=False) -> float:
        """Compute epsilon for approximate differential privacy with given delta.
        
        amplify=True
                Get privacy amplification by subsampling

        verbose=True
                   See search progress report            
        """
        def candidate_adp_epsilon(rdp_order: float|int):
            assert rdp_order > 1

            # See [DDG-J] Lemma 4 
            # Note (epsilon^2)/2 - concentrated DP is equivalent to (alpha, alpha/2 * epsilon^2) - RDP for all alpha > 1
            # so the first summand in [DDG-J] Lemma 4 is actually the RDP-epsilon given by self.fl_rdp
            return self.fl_rdp(rdp_order, amplify) + log(1/(rdp_order*self.delta)) / (rdp_order-1) + log(1-1/rdp_order)
        
        optimal_adp_epsilon = float('inf')

        if amplify:
            for rdp_order in range(2, 31):
                candidate = candidate_adp_epsilon(rdp_order)
                if candidate < optimal_adp_epsilon:
                    optimal_adp_epsilon = candidate
                    # for debugging 
                    if verbose:
                        print(f'(alpha, epsilon) = ({rdp_order}, {optimal_adp_epsilon})')

        if not amplify:
            interval_length = 0.01
            upper_bound = 10

            current_rdp_order = 1
            while current_rdp_order < upper_bound:
                current_rdp_order += interval_length
                candidate = candidate_adp_epsilon(current_rdp_order)
                if candidate < optimal_adp_epsilon:
                    optimal_adp_epsilon = candidate
                    # for debugging 
                    if verbose:
                        print(f'(alpha, epsilon) = ({current_rdp_order}, {optimal_adp_epsilon})')

        return optimal_adp_epsilon
    
    def __repr__(self) -> str:

        rdp_order = 2
        verbose = False
        ndecimals = 2

        single_round_cdp = round(accountant.single_fl_round_concentrated_dp(), ndecimals)
        single_round_rdp = round(accountant.single_fl_round_renyi_dp(rdp_order), ndecimals)

        fl_rdp_unamplified = round(accountant.fl_rdp(rdp_order=rdp_order, amplify=False), ndecimals)
        fl_adp_unamplified = round(accountant.fl_approximate_dp(amplify=False, verbose=verbose), ndecimals)

        try:
            # amplified rdp requires rdp_order to be an integer
            fl_rdp_amplified = round(accountant.fl_rdp(rdp_order=int(rdp_order), amplify=True), ndecimals)
            fl_adp_amplified = round(accountant.fl_approximate_dp(amplify=True, verbose=verbose), ndecimals)

            report = f"""
                single_round_cdp = {single_round_cdp}
                (alpha, single_round_rdp) = ({rdp_order}, {single_round_rdp})
                (alpha, fl_rdp_unamplified) = ({int(rdp_order)}, {fl_rdp_unamplified})
                (alpha, fl_rdp_amplified) = ({rdp_order}, {fl_rdp_amplified})
                (fl_adp_unamplified, delta) = ({fl_adp_unamplified}, {self.delta})
                (fl_adp_amplified, delta) = ({fl_adp_amplified}, {self.delta})
            """
        except OverflowError:
            report = f"""
                single_round_cdp = {single_round_cdp}
                (alpha, single_round_rdp) = ({rdp_order}, {single_round_rdp})
                (alpha, fl_rdp_unamplified) = ({int(rdp_order)}, {fl_rdp_unamplified})
                (alpha, fl_rdp_amplified) = (OverflowError)
                (fl_adp_unamplified, delta) = ({fl_adp_unamplified}, {self.delta})
                (fl_adp_amplified, delta) = (OverflowError)
            """

        return report

if __name__ == '__main__':

    # EMNIST
    parameters = {
        'l2_norm_clip': 0.03,
        'private_vector_dimension': 1_018_174,
        'granularity': 3.5e-6, 
        'noise_scale': 9.5e-4, 
        'n_trustworthy_clients': 3400,
        'n_fl_rounds': 1500,
        'privacy_amplification_sampling_ratio': 1/34,
        'bias': exp(-0.5), 
        'approximate_dp_delta': 1/3400
    }

    print('EMNIST (DDGauss paper)')
    accountant = DDGaussAccountant(**parameters)
    print(accountant)

    # Fed-ISIC2019
    isic_train_size = int(0.8 * 23247)
    isic_batch_size = 64

    parameters = {
        'l2_norm_clip': 1,
        'private_vector_dimension': 28406122,
        'granularity': 1e-6, 
        'noise_scale': 1.212e-3, 
        'n_trustworthy_clients': 3,
        'n_fl_rounds': 100,
        'privacy_amplification_sampling_ratio': isic_batch_size/isic_train_size,
        'bias': exp(-0.5), 
        'approximate_dp_delta': isic_train_size**-2
    }

    print('Fed-ISIC2019')
    accountant = DDGaussAccountant(**parameters)
    print(accountant)
