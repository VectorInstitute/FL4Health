from flwr.common.typing import (
    Code,
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


class CustomClientProxy(ClientProxy):
    """Subclass of ClientProxy."""

    def __init__(self, cid: str, num_samples: int = 1):
        super().__init__(cid)
        self.properties = {"num_samples": num_samples}

    def get_properties(
        self,
        ins: GetPropertiesIns,
        timeout: float | None,
        group_id: int | None,
    ) -> GetPropertiesRes:
        status: Status = Status(code=Code["OK"], message="Test")
        return GetPropertiesRes(status=status, properties=self.properties)

    def get_parameters(
        self,
        ins: GetParametersIns,
        timeout: float | None,
        group_id: int | None,
    ) -> GetParametersRes:
        raise NotImplementedError

    def fit(
        self,
        ins: FitIns,
        timeout: float | None,
        group_id: int | None,
    ) -> FitRes:
        raise NotImplementedError

    def evaluate(
        self,
        ins: EvaluateIns,
        timeout: float | None,
        group_id: int | None,
    ) -> EvaluateRes:
        raise NotImplementedError

    def reconnect(
        self,
        ins: ReconnectIns,
        timeout: float | None,
        group_id: int | None,
    ) -> DisconnectRes:
        raise NotImplementedError
