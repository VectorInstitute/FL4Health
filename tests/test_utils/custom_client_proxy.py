from typing import Optional

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
        timeout: Optional[float],
        group_id: Optional[int],
    ) -> GetPropertiesRes:
        status: Status = Status(code=Code["OK"], message="Test")
        res = GetPropertiesRes(status=status, properties=self.properties)
        return res

    def get_parameters(
        self,
        ins: GetParametersIns,
        timeout: Optional[float],
        group_id: Optional[int],
    ) -> GetParametersRes:
        raise NotImplementedError

    def fit(
        self,
        ins: FitIns,
        timeout: Optional[float],
        group_id: Optional[int],
    ) -> FitRes:
        raise NotImplementedError

    def evaluate(
        self,
        ins: EvaluateIns,
        timeout: Optional[float],
        group_id: Optional[int],
    ) -> EvaluateRes:
        raise NotImplementedError

    def reconnect(
        self,
        ins: ReconnectIns,
        timeout: Optional[float],
        group_id: Optional[int],
    ) -> DisconnectRes:
        raise NotImplementedError
