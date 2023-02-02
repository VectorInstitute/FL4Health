from typing import List, Optional, Union

from flwr.server.client_manager import SimpleClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.criterion import Criterion


class BaseSamplingManager(SimpleClientManager):
    """Overrides the Simple Client Manager to Provide Fixed Sampling without replacement for Clients"""

    def sample(
        self,
        sample_fraction: float,
        min_num_clients: Optional[int] = None,
        criterion: Optional[Criterion] = None,
    ) -> List[ClientProxy]:
        raise NotImplementedError

    def wait_for_min_num_clients(self, min_num_clients: Union[int, None]) -> None:
        if min_num_clients is not None:
            self.wait_for(min_num_clients)
        else:
            self.wait_for(1)

    def sample_all(
        self, min_num_clients: Optional[int] = None, criterion: Optional[Criterion] = None
    ) -> List[ClientProxy]:

        self.wait_for_min_num_clients(min_num_clients)

        available_cids = list(self.clients)
        if criterion is not None:
            available_cids = [cid for cid in available_cids if criterion.select(self.clients[cid])]

        return [self.clients[cid] for cid in available_cids]
