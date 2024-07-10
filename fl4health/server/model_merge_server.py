import timeit
from typing import Optional, Tuple
from logging import INFO, DEBUG

from flwr.server.server import Server
from flwr.common.logger import log
from flwr.server.history import History
from flwr.server.strategy import Strategy


class ModelMergeServer(Server):
    def fit(self, num_rounds: int, timeout: Optional[float]) -> Tuple[History, float]:
        """

        Args:
            num_rounds (int): Not used.
            timeout (Optional[float]): Timeout in seconds that the server should wait for the clients to response.
                If none, then it will wait for the minimum number to respond indefinitely.

        Returns:
            Tuple[History, float]: The first element of the tuple is a History object containing the aggregated
                metrics returned from the clients. Tuple also contains elapsed time in seconds for round.
        """
        history = History()

        # Run Federated Model Merging
        log(INFO, "Federated Model Merging Starting")
        start_time = timeit.default_timer()

        res_fit = self.fit_round(
            server_round=1,
            timeout=timeout,
        )

        if res_fit is not None:
            parameters_prime, _, _ = res_fit  # fit_metrics_aggregated
            if parameters_prime:
                self.parameters = parameters_prime
            else:
                log(DEBUG, "Federated Model Merging Failed")

        # Bookkeeping
        end_time = timeit.default_timer()
        elapsed = end_time - start_time
        log(INFO, "Federated Model Merging Finished in %s", elapsed)
        return history, elapsed
