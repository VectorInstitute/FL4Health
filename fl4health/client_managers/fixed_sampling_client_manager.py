from flwr.server.client_manager import SimpleClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.criterion import Criterion


class FixedSamplingClientManager(SimpleClientManager):
    """Keeps sampling fixed until it's reset"""

    def __init__(self) -> None:
        super().__init__()
        self.current_sample: list[ClientProxy] | None = None

    def reset_sample(self) -> None:
        """Resets the saved sample so self.sample produces a new sample again."""
        self.current_sample = None

    def sample(
        self,
        num_clients: int,
        min_num_clients: int | None = None,
        criterion: Criterion | None = None,
    ) -> list[ClientProxy]:
        """
        Return a new client sample for the first time it runs. For subsequent runs,
        it will return the same sampling until self.reset_sampling() is called.

        Args:
            num_clients: (int) The number of clients to sample.
            min_num_clients: (int | None) The minimum number of clients to return in the sample.
                Optional, default is num_clients.
            criterion: (Criterion | None) A criterion to filter clients to sample.
                Optional, default is no criterion (no filter).

        Returns:
            list[ClientProxy]: A list of sampled clients as ClientProxy instances.
        """
        if self.current_sample is None:
            self.current_sample = super().sample(num_clients, min_num_clients, criterion)
        return self.current_sample
