from logging import INFO
from typing import List, Optional

from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.history import History
from flwr.server.server import Server
from flwr.server.strategy import Strategy

from fl4health.reporting.fl_wanb import ServerWandBReporter
from fl4health.server.polling import poll_clients
from fl4health.strategies.client_dp_fedavgm import ClientLevelDPFedAvgM


class FlServer(Server):
    """
    Base Server for the library to facilitate strapping additional/userful machinery to the base flwr server.
    """

    def __init__(
        self,
        client_manager: ClientManager,
        strategy: Optional[Strategy] = None,
        wandb_reporter: Optional[ServerWandBReporter] = None,
    ) -> None:
        super().__init__(client_manager=client_manager, strategy=strategy)
        self.wandb_reporter = wandb_reporter

    def fit(self, num_rounds: int, timeout: Optional[float]) -> History:
        history = super().fit(num_rounds, timeout)
        if self.wandb_reporter:
            # report history to W and B
            self.wandb_reporter.report_metrics(num_rounds, history)
        return history

    def shutdown(self) -> None:
        if self.wandb_reporter:
            self.wandb_reporter.shutdown_reporter()


class ClientLevelDPWeightedFedAvgServer(Server):
    """
    Server to be used in case of Client Level Differential Privacy with weighted Federated Averaging.
    Modified the fit function to poll clients for sample counts prior to the first round of FL.
    """

    def __init__(self, *, client_manager: ClientManager, strategy: ClientLevelDPFedAvgM) -> None:
        super().__init__(client_manager=client_manager, strategy=strategy)

    def fit(self, num_rounds: int, timeout: Optional[float]) -> History:
        """Run federated averaging for a number of rounds."""

        # Poll clients for sample counts
        log(INFO, "Polling Clients for sample counts")
        assert isinstance(self.strategy, ClientLevelDPFedAvgM)
        client_instructions = self.strategy.configure_poll(server_round=1, client_manager=self._client_manager)
        results, _ = poll_clients(
            client_instructions=client_instructions, max_workers=self.max_workers, timeout=timeout
        )

        sample_counts: List[int] = [
            int(get_properties_res.properties["num_samples"]) for (_, get_properties_res) in results
        ]

        # If Weighted FedAvg, set sample counts to compute client weights
        if self.strategy.weighted_averaging:
            self.strategy.sample_counts = sample_counts

        return super().fit(num_rounds=num_rounds, timeout=timeout)
