from logging import WARNING
from typing import List, Optional

import numpy as np
from flwr.common.logger import log
from flwr.server.client_proxy import ClientProxy
from flwr.server.criterion import Criterion

from fl4health.client_managers.base_sampling_manager import BaseFractionSamplingManager


class PoissonSamplingClientManager(BaseFractionSamplingManager):
    """Overrides the Simple Client Manager to Provide Poisson Sampling for Clients rather than
    fixed without replacement sampling"""

    def _poisson_sample(self, sampling_probability: float, available_cids: List[str]) -> List[str]:
        poisson_trials = np.random.binomial(1, sampling_probability, len(available_cids))
        poisson_mask = poisson_trials.astype(dtype=bool)
        return list(np.array(available_cids)[poisson_mask])

    def sample_fraction(
        self,
        sample_fraction: float,
        min_num_clients: Optional[int] = None,
        criterion: Optional[Criterion] = None,
    ) -> List[ClientProxy]:
        """Poisson Sampling of Flower ClientProxy instances with a probability determine by sample_fraction."""

        available_cids = self.wait_and_filter(min_num_clients, criterion)
        n_available_cids = len(available_cids)
        expected_clients_selected = sample_fraction * n_available_cids
        if expected_clients_selected < 1:
            log(
                WARNING,
                f"Sample fraction of {round(sample_fraction, 3)} from {n_available_cids} clients results "
                f"in expected value of {round(expected_clients_selected, 3)} selected.",
            )
        sampled_cids = self._poisson_sample(sample_fraction, available_cids)
        return [self.clients[cid] for cid in sampled_cids]
