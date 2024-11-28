from typing import Union

from flwr.server.client_manager import SimpleClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.criterion import Criterion


class BaseFractionSamplingManager(SimpleClientManager):
    """Overrides the Simple Client Manager to Provide Fixed Sampling without replacement for Clients"""

    def sample(
        self, num_clients: int, min_num_clients: int | None = None, criterion: Criterion | None = None
    ) -> list[ClientProxy]:
        raise NotImplementedError(
            "The basic sampling function is not implemented for these managers. "
            "Please use the fraction sample function instead"
        )

    def sample_fraction(
        self,
        sample_fraction: float,
        min_num_clients: int | None = None,
        criterion: Criterion | None = None,
    ) -> list[ClientProxy]:
        raise NotImplementedError

    def wait_and_filter(self, min_num_clients: Union[int, None], criterion: Criterion | None = None) -> list[str]:
        if min_num_clients is not None:
            self.wait_for(min_num_clients)
        else:
            self.wait_for(1)

        available_cids = list(self.clients)
        if criterion is not None:
            available_cids = [cid for cid in available_cids if criterion.select(self.clients[cid])]

        return available_cids

    def sample_all(self, min_num_clients: int | None = None, criterion: Criterion | None = None) -> list[ClientProxy]:
        available_cids = self.wait_and_filter(min_num_clients, criterion)

        return [self.clients[cid] for cid in available_cids]
