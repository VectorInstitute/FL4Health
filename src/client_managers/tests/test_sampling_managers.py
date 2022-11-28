from typing import Optional

import numpy as np
import pytest
from flwr.common import (
    DisconnectRes,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    GetPropertiesIns,
    GetPropertiesRes,
    ReconnectIns,
)
from flwr.server.client_proxy import ClientProxy

from src.client_managers.fixed_without_replacement_manager import FixedSamplingWithoutReplacementClientManager
from src.client_managers.poisson_sampling_manager import PoissonSamplingClientManager


class CustomClientProxy(ClientProxy):
    """Subclass of ClientProxy."""

    def get_properties(
        self,
        ins: GetPropertiesIns,
        timeout: Optional[float],
    ) -> GetPropertiesRes:
        """Returns the client's properties."""

    def get_parameters(
        self,
        ins: GetParametersIns,
        timeout: Optional[float],
    ) -> GetParametersRes:
        """Return the current local model parameters."""

    def fit(
        self,
        ins: FitIns,
        timeout: Optional[float],
    ) -> FitRes:
        """Refine the provided weights using the locally held dataset."""

    def evaluate(
        self,
        ins: EvaluateIns,
        timeout: Optional[float],
    ) -> EvaluateRes:
        """Evaluate the provided weights using the locally held dataset."""

    def reconnect(
        self,
        ins: ReconnectIns,
        timeout: Optional[float],
    ) -> DisconnectRes:
        """Disconnect and (optionally) reconnect later."""


def test_poisson_sampling_subset() -> None:
    np.random.seed(42)
    client_manager = PoissonSamplingClientManager()
    available_cids = ["c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9", "c10"]
    sample = client_manager._poisson_sample(0.3, available_cids)
    print(sample)
    expected_sublist = ["c2", "c3", "c8", "c10"]
    assert len(expected_sublist) == len(sample)
    assert all([a == b for a, b in zip(expected_sublist, sample)])


def test_poisson_sampling_when_low_probability(caplog: pytest.LogCaptureFixture) -> None:
    np.random.seed(42)
    client_manager = PoissonSamplingClientManager()
    client_proxies = [
        CustomClientProxy("c1"),
        CustomClientProxy("c2"),
        CustomClientProxy("c3"),
        CustomClientProxy("c4"),
        CustomClientProxy("c5"),
        CustomClientProxy("c6"),
        CustomClientProxy("c7"),
    ]
    for client_proxy in client_proxies:
        client_manager.register(client_proxy)
    sample = client_manager.sample(0.01, 2)
    assert "WARNING  flower:poisson_sampling_manager.py" in caplog.text
    assert len(sample) == 1


def test_fixed_without_replacement_subset() -> None:
    np.random.seed(42)
    client_manager = FixedSamplingWithoutReplacementClientManager()
    client_proxies = [
        CustomClientProxy("c1"),
        CustomClientProxy("c2"),
        CustomClientProxy("c3"),
        CustomClientProxy("c4"),
        CustomClientProxy("c5"),
        CustomClientProxy("c6"),
        CustomClientProxy("c7"),
        CustomClientProxy("c8"),
        CustomClientProxy("c9"),
        CustomClientProxy("c10"),
        CustomClientProxy("c11"),
    ]
    for client_proxy in client_proxies:
        client_manager.register(client_proxy)
    sample = client_manager.sample(0.3, 2)
    assert len(sample) == 3


def test_fixed_sampling_when_low_probability(caplog: pytest.LogCaptureFixture) -> None:
    np.random.seed(42)
    client_manager = FixedSamplingWithoutReplacementClientManager()
    client_proxies = [
        CustomClientProxy("c1"),
        CustomClientProxy("c2"),
        CustomClientProxy("c3"),
        CustomClientProxy("c4"),
        CustomClientProxy("c5"),
        CustomClientProxy("c6"),
        CustomClientProxy("c7"),
    ]
    for client_proxy in client_proxies:
        client_manager.register(client_proxy)
    sample = client_manager.sample(0.01, 2)
    assert "WARNING  flower:fixed_without_replacement_manager.py" in caplog.text
    assert len(sample) == 1
