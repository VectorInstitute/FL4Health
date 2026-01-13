import random
from logging import WARNING

from flwr.common.logger import log
from flwr.server.client_proxy import ClientProxy
from flwr.server.criterion import Criterion

from fl4health.client_managers.base_sampling_manager import BaseFractionSamplingManager


class FixedSamplingByFractionClientManager(BaseFractionSamplingManager):
    """
    Overrides the ``BaseFractionSamplingManager`` to provide Fixed Sampling without replacement for Clients by
    fraction.
    """

    def sample_fraction(
        self,
        sample_fraction: float,
        min_num_clients: int | None = None,
        criterion: Criterion | None = None,
    ) -> list[ClientProxy]:
        """
        Sample a number of Flower ``ClientProxy`` instances **WITHOUT** replacement.

        Args:
            sample_fraction (float): Fraction of clients to sample. Guaranteed to produce this fraction from
                available clients
            min_num_clients (int | None, optional): minimum number of clients required to be available.
                Defaults to None.
            criterion (Criterion | None, optional): Criterion to help with sampling. No filter is applied if None.
                Defaults to None.

        Returns:
            (list[ClientProxy]): List of ClientProxy objects representing the selected clients.
        """
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
