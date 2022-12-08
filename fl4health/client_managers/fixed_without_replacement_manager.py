import random
from logging import WARNING
from typing import List, Optional

from flwr.common.logger import log
from flwr.server.client_proxy import ClientProxy
from flwr.server.criterion import Criterion

from fl4health.client_managers.base_sampling_manager import BaseSamplingManager


class FixedSamplingWithoutReplacementClientManager(BaseSamplingManager):
    """Overrides the Simple Client Manager to Provide Fixed Sampling without replacement for Clients"""

    def sample(
        self,
        sample_fraction: float,
        # minimum number of clients required to be available
        min_num_clients: Optional[int] = None,
        criterion: Optional[Criterion] = None,
    ) -> List[ClientProxy]:
        """Sample a number of Flower ClientProxy instances."""
        if min_num_clients is not None:
            self.wait_for(min_num_clients)
        else:
            # Wait for at least one client to be available
            self.wait_for(1)
        # Sample clients which meet the criterion
        available_cids = list(self.clients)
        if criterion is not None:
            available_cids = [cid for cid in available_cids if criterion.select(self.clients[cid])]

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
