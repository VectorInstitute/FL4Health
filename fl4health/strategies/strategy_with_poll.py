from abc import ABC, abstractmethod
from typing import List, Tuple

from flwr.common import GetPropertiesIns
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy


class StrategyWithPolling(ABC):
    """
    This abstract base class is used to ensure that an FL strategy class implements configure polling when it should
    and that any server that wants to do polling can use this function when it's expected to.
    """

    @abstractmethod
    def configure_poll(
        self, server_round: int, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, GetPropertiesIns]]:
        pass
