import random

from flwr.server.client_manager import SimpleClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.criterion import Criterion


class BaseFractionSamplingManager(SimpleClientManager):
    """Overrides the ``SimpleClientManager`` to Provide Fixed Sampling without replacement for Clients."""

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

    def wait_and_filter(self, min_num_clients: int | None, criterion: Criterion | None = None) -> list[str]:
        """
        Waits for ``min_num_clients`` to become available then select clients from those available and filter them
        based on the criterion provided. If ``min_num_clients`` is None, then it waits for at least 1 client to be
        available.

        Args:
            min_num_clients (int | None): Number of clients to wait for before performing filtration.
            criterion (Criterion | None, optional): criterion used to filter available clients. Defaults to None.

        Returns:
            (list[str]): List of CIDs representing available and filtered clients.
        """
        if min_num_clients is not None:
            self.wait_for(min_num_clients)
        else:
            self.wait_for(1)

        available_cids = list(self.clients)
        if criterion is not None:
            available_cids = [cid for cid in available_cids if criterion.select(self.clients[cid])]

        return available_cids

    def sample_one(self, min_num_clients: int | None = None, criterion: Criterion | None = None) -> list[ClientProxy]:
        """
        Samples exactly one available client randomly. This should only be used for client-side parameter
        initialization.

        Args:
            min_num_clients (int | None, optional): minimum number of clients to wait to become available before
                selecting all available clients. Defaults to None.
            criterion (Criterion | None, optional): Criterion used to filter returned clients. If none, no filter is
                applied. Defaults to None.

        Returns:
            (list[ClientProxy]): Selected client represented by a ClientProxy object in list form as expected by
                server.
        """
        available_cids = self.wait_and_filter(min_num_clients, criterion)
        # Sample exactly on client randomly
        cids = random.sample(available_cids, 1)

        return [self.clients[cid] for cid in cids]

    def sample_all(self, min_num_clients: int | None = None, criterion: Criterion | None = None) -> list[ClientProxy]:
        """
        Samples **ALL** available clients.

        Args:
            min_num_clients (int | None, optional): minimum number of clients to wait to become available before
                selecting all available clients. Defaults to None.
            criterion (Criterion | None, optional): Criterion used to filter returned clients. If none, no filter is
                applied. Defaults to None.

        Returns:
            (list[ClientProxy]): List of selected clients represented by ``ClientProxy`` objects.
        """
        available_cids = self.wait_and_filter(min_num_clients, criterion)

        return [self.clients[cid] for cid in available_cids]
