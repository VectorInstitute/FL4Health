import random
from logging import WARNING
from typing import List, Optional

from flwr.common.logger import log
from flwr.server.client_proxy import ClientProxy
from flwr.server.criterion import Criterion

from fl4health.client_managers.base_sampling_manager import BaseFractionSamplingManager


class FixedSamplingByFractionClientManager(BaseFractionSamplingManager):
    """Overrides the Simple Client Manager to provide Fixed Sampling without replacement for Clients by fraction"""

    def sample_fraction(
        self,
        sample_fraction: float,
        # minimum number of clients required to be available
        min_num_clients: Optional[int] = None,
        criterion: Optional[Criterion] = None,
    ) -> List[ClientProxy]:
        """Sample a number of Flower ClientProxy instances."""

        available_cids = self.wait_and_filter(min_num_clients, criterion)
        n_available_cids = len(available_cids)
        num_to_sample = int(sample_fraction * n_available_cids)
        if num_to_sample < 1:
            log(
                WARNING,
                f"Sample fraction of {round(sample_fraction, 3)} resulted in 0 samples to being selected"
                f"from {n_available_cids}.",
            )
            return []
        sampled_cids = random.sample(available_cids, num_to_sample)
        return [self.clients[cid] for cid in sampled_cids]
