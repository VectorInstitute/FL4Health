from abc import ABC, abstractmethod
from typing import List, Tuple

from flwr.common import GetPropertiesIns
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy


class StrategyWithPolling(ABC):
    @abstractmethod
    def configure_poll(
        self, server_round: int, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, GetPropertiesIns]]:
        pass
