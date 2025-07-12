import numpy as np
import pytest
from flwr.common import NDArrays, Parameters

from fl4health.strategies.client_dp_fedavgm import ClientLevelDPFedAvgM


def generate_clipping_bits(n_bits: int, clipping_bound: float, mean: float, std_dev: float) -> NDArrays:
    np.random.seed(42)
    generated_grad_values = np.random.normal(mean, std_dev, size=(n_bits))
    boolean_np_array = (np.exp(generated_grad_values) < clipping_bound).astype(int)
    return [np.array(e) for e in boolean_np_array.tolist()]


def test_adaptive_clipping_convergence() -> None:
    np.random.seed(42)
    # Test reproducing the adaptive clipping convergence estimates of Figure 2 from Differentially Private Learning
    # with Adaptive Clipping
    num_clients_sampled = 100
    clipping_noise_multiplier = num_clients_sampled / 20.0
    initial_clipping_bound = 0.1
    clipping_learning_rate = 0.2

    # Quartile 0.5, gradient magnitude drawn from e^(N(0.0, 1.0))
    strategy = ClientLevelDPFedAvgM(
        initial_parameters=Parameters([], ""),
        adaptive_clipping=True,
        initial_clipping_bound=initial_clipping_bound,
        clipping_learning_rate=clipping_learning_rate,
        clipping_quantile=0.5,
        clipping_noise_multiplier=clipping_noise_multiplier,
    )
    clipping_values = [strategy.clipping_bound]
    for _ in range(200):
        clipping_bound_t = strategy.clipping_bound
        clipping_bits = generate_clipping_bits(num_clients_sampled, clipping_bound_t, 0.0, 1.0)
        strategy.update_clipping_bound(clipping_bits)
        clipping_values.append(strategy.clipping_bound)

    assert pytest.approx(strategy.clipping_bound, abs=0.1) == 1.0

    # Quartile 0.7, gradient magnitude drawn from e^(N(ln 10, 1.0))
    strategy = ClientLevelDPFedAvgM(
        initial_parameters=Parameters([], ""),
        adaptive_clipping=True,
        initial_clipping_bound=initial_clipping_bound,
        clipping_learning_rate=clipping_learning_rate,
        clipping_quantile=0.7,
        clipping_noise_multiplier=clipping_noise_multiplier,
    )

    clipping_values = [strategy.clipping_bound]
    for _ in range(200):
        clipping_bound_t = strategy.clipping_bound
        clipping_bits = generate_clipping_bits(num_clients_sampled, clipping_bound_t, np.log(10), 1.0)
        strategy.update_clipping_bound(clipping_bits)
        clipping_values.append(strategy.clipping_bound)

    # NOTE: that this is a much "noisier" estimate because the clipping bound is farther from the initial
    # guess. Thus the abs of 1.0
    assert pytest.approx(strategy.clipping_bound, abs=1.0) == 17.5
