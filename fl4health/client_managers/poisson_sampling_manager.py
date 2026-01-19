from logging import WARNING

import numpy as np
from flwr.common.logger import log
from flwr.server.client_proxy import ClientProxy
from flwr.server.criterion import Criterion

from fl4health.client_managers.base_sampling_manager import BaseFractionSamplingManager


class PoissonSamplingClientManager(BaseFractionSamplingManager):
    """
    Overrides the ``BaseFractionSamplingManager`` to provide Poisson sampling for clients rather than fixed without
    replacement sampling.
    """

    def _poisson_sample(self, sampling_probability: float, available_cids: list[str]) -> list[str]:
        poisson_trials = np.random.binomial(1, sampling_probability, len(available_cids))
        poisson_mask = poisson_trials.astype(dtype=bool)
        return list(np.array(available_cids)[poisson_mask])

    def sample_fraction(
        self,
        sample_fraction: float,
        min_num_clients: int | None = None,
        criterion: Criterion | None = None,
    ) -> list[ClientProxy]:
        """
        Poisson Sampling of Flower ClientProxy instances with a probability determine by sample_fraction.

        Args:
            sample_fraction (float): Fraction, which sets the Poisson sampling probability
            min_num_clients (int | None, optional): minimum number of clients to be selected (overrides sampling to
                some extent). Defaults to None.
            criterion (Criterion | None, optional): Criterion to sample clients based on. Defaults to None.

        Returns:
            (list[ClientProxy]): List of selected ClientProxy objects represented the clients selected by the process.
        """
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
