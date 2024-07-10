import datetime
import timeit
from logging import DEBUG, INFO
from typing import Optional, Tuple

from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.history import History
from flwr.server.server import Server
from flwr.server.strategy import Strategy

from fl4health.reporting.metrics import MetricsReporter
from fl4health.strategies.model_merge_strategy import ModelMergeStrategy


class ModelMergeServer(Server):
    def __init__(
        self,
        *,
        client_manager: ClientManager,
        strategy: Optional[Strategy] = None,
        metrics_reporter: Optional[MetricsReporter] = None,
    ) -> None:
        assert isinstance(strategy, ModelMergeStrategy)
        super().__init__(client_manager=client_manager, strategy=strategy)

        if metrics_reporter is not None:
            self.metrics_reporter = metrics_reporter
        else:
            self.metrics_reporter = MetricsReporter()

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
        self.metrics_reporter.add_to_metrics({"type": "server", "fit_start": datetime.datetime.now()})

        history = History()

        # Run Federated Model Merging
        log(INFO, "Federated Model Merging Starting")
        start_time = timeit.default_timer()

        res_fit = self.fit_round(
            server_round=1,
            timeout=timeout,
        )

        if res_fit is not None:
            parameters_prime, fit_metrics, _ = res_fit  # fit_metrics_aggregated
            if parameters_prime:
                self.parameters = parameters_prime
            history.add_metrics_distributed_fit(server_round=1, metrics=fit_metrics)
        else:
            log(DEBUG, "Federated Model Merging Failed")

        res_fed = self.evaluate_round(server_round=1, timeout=timeout)
        if res_fed is not None:
            _, evaluate_metrics_fed, _ = res_fed
            if evaluate_metrics_fed is not None:
                history.add_metrics_distributed(server_round=1, metrics=evaluate_metrics_fed)

        self.metrics_reporter.add_to_metrics(
            data={
                "fit_end": datetime.datetime.now(),
                "metrics_centralized": history.metrics_centralized,
                "losses_centralized": history.losses_centralized,
            }
        )

        # Bookkeeping
        end_time = timeit.default_timer()
        elapsed = end_time - start_time
        log(INFO, "Federated Model Merging Finished in %s", elapsed)
        return history, elapsed
