import numpy as np
import pytest

from fl4health.client_managers.base_sampling_manager import BaseSamplingManager
from fl4health.client_managers.fixed_without_replacement_manager import FixedSamplingWithoutReplacementClientManager
from fl4health.client_managers.poisson_sampling_manager import PoissonSamplingClientManager
from tests.client_managers.fixtures import create_and_register_clients_to_manager  # noqa


def test_poisson_sampling_subset() -> None:  # noqa
    np.random.seed(42)
    client_manager = PoissonSamplingClientManager()
    available_cids = ["c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9", "c10"]
    sample = client_manager._poisson_sample(0.3, available_cids)
    expected_sublist = ["c2", "c3", "c8", "c10"]
    assert len(expected_sublist) == len(sample)
    assert all([a == b for a, b in zip(expected_sublist, sample)])


@pytest.mark.parametrize("client_manager,num_clients", [(PoissonSamplingClientManager(), 7)])
def test_poisson_sampling_when_low_probability(
    caplog: pytest.LogCaptureFixture, create_and_register_clients_to_manager: BaseSamplingManager  # noqa
) -> None:
    np.random.seed(42)
    client_manager = create_and_register_clients_to_manager
    sample = client_manager.sample_fraction(0.01, 2)
    assert "WARNING  flwr:poisson_sampling_manager.py" in caplog.text
    assert len(sample) == 0


@pytest.mark.parametrize("client_manager,num_clients", [(FixedSamplingWithoutReplacementClientManager(), 11)])
def test_fixed_without_replacement_subset(create_and_register_clients_to_manager: BaseSamplingManager) -> None:  # noqa
    np.random.seed(42)
    client_manager = create_and_register_clients_to_manager
    sample = client_manager.sample_fraction(0.3, 2)
    assert len(sample) == 3


@pytest.mark.parametrize("client_manager,num_clients", [(FixedSamplingWithoutReplacementClientManager(), 7)])
def test_fixed_sampling_when_low_probability(
    caplog: pytest.LogCaptureFixture, create_and_register_clients_to_manager: BaseSamplingManager  # noqa
) -> None:
    np.random.seed(42)
    client_manager = create_and_register_clients_to_manager
    sample = client_manager.sample_fraction(0.01, 2)
    assert "WARNING  flwr:fixed_without_replacement_manager.py" in caplog.text
    assert len(sample) == 0
