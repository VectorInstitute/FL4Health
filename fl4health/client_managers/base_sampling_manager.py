from typing import List, Optional

from flwr.server.client_manager import SimpleClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.criterion import Criterion


class BaseSamplingManager(SimpleClientManager):
    """Overrides the Simple Client Manager to Provide Fixed Sampling without replacement for Clients"""

    def sample(
        self, num_clients: int, min_num_clients: Optional[int] = None, criterion: Optional[Criterion] = None
    ) -> List[ClientProxy]:
        raise NotImplementedError(
            "The basic sampling function is not implemented for these managers. "
            "Please use the fraction sample function instead"
        )

    def sample_fraction(
        self,
        sample_fraction: float,
        min_num_clients: Optional[int] = None,
        criterion: Optional[Criterion] = None,
    ) -> List[ClientProxy]:
        raise NotImplementedError
