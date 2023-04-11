from typing import List, Optional, Tuple

from flwr.common.typing import (
    Code,
    Config,
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
    Status,
)
from flwr.server.client_proxy import ClientProxy

from fl4health.server.server import poll_client, poll_clients


class CustomClientProxy(ClientProxy):
    """Subclass of ClientProxy."""

    def __init__(self, cid: str, num_samples: int = 1):
        super().__init__(cid)
        self.properties = {"num_samples": num_samples}

    def get_properties(
        self,
        ins: GetPropertiesIns,
        timeout: Optional[float],
    ) -> GetPropertiesRes:
        status: Status = Status(code=Code["OK"], message="Test")
        res = GetPropertiesRes(status=status, properties=self.properties)
        return res

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


def test_poll_client() -> None:
    client_proxy = CustomClientProxy(cid="c0", num_samples=10)
    config: Config = {"test": 0}

    ins = GetPropertiesIns(config=config)
    _, res = poll_client(client_proxy, ins)

    assert res.properties["num_samples"] == 10


def test_poll_clients() -> None:
    client_ids = [f"c{i}" for i in range(10)]
    sample_counts = [11 for _ in range(10)]
    clients = [CustomClientProxy(cid, count) for cid, count in zip(client_ids, sample_counts)]
    config: Config = {"test": 0}
    ins = GetPropertiesIns(config=config)
    clients_instructions: List[Tuple[ClientProxy, GetPropertiesIns]] = [(client, ins) for client in clients]

    results, failures = poll_clients(client_instructions=clients_instructions, max_workers=None, timeout=None)

    property_results = [result[1] for result in results]

    for res in property_results:
        assert res.properties["num_samples"] == 11
