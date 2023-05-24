from typing import Optional

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

from fl4health.client_managers.base_sampling_manager import BaseSamplingManager


class CustomClientProxy(ClientProxy):
    """Subclass of ClientProxy."""

    def get_properties(
        self,
        ins: GetPropertiesIns,
        timeout: Optional[float],
    ) -> GetPropertiesRes:
        raise NotImplementedError

    def get_parameters(
        self,
        ins: GetParametersIns,
        timeout: Optional[float],
    ) -> GetParametersRes:
        raise NotImplementedError

    def fit(
        self,
        ins: FitIns,
        timeout: Optional[float],
    ) -> FitRes:
        raise NotImplementedError

    def evaluate(
        self,
        ins: EvaluateIns,
        timeout: Optional[float],
    ) -> EvaluateRes:
        raise NotImplementedError

    def reconnect(
        self,
        ins: ReconnectIns,
        timeout: Optional[float],
    ) -> DisconnectRes:
        raise NotImplementedError


@pytest.fixture
def create_and_register_clients_to_manager(
    client_manager: BaseSamplingManager, num_clients: int
) -> BaseSamplingManager:
    client_proxies = [CustomClientProxy(f"c{str(i)}") for i in range(1, num_clients + 1)]

    for client in client_proxies:
        client_manager.register(client)

    return client_manager
