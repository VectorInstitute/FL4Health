from logging import INFO
from typing import List, Optional

from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.history import History
from flwr.server.server import Server
from flwr.server.strategy import Strategy

from fl4health.checkpointing.checkpointer import TorchCheckpointer
from fl4health.reporting.fl_wanb import ServerWandBReporter
from fl4health.server.polling import poll_clients
from fl4health.strategies.fedavg_sampling import FedAvgSampling


class FlServer(Server):
    """
    Base Server for the library to facilitate strapping additional/userful machinery to the base flwr server.
    """

    def __init__(
        self,
        client_manager: ClientManager,
        strategy: Optional[Strategy] = None,
        wandb_reporter: Optional[ServerWandBReporter] = None,
        checkpointer: Optional[TorchCheckpointer] = None,
    ) -> None:
        super().__init__(client_manager=client_manager, strategy=strategy)
        self.wandb_reporter = wandb_reporter
        self.checkpointer = checkpointer

    def fit(self, num_rounds: int, timeout: Optional[float]) -> History:
        history = super().fit(num_rounds, timeout)
        if self.wandb_reporter:
            # report history to W and B
            self.wandb_reporter.report_metrics(num_rounds, history)
        return history

    def shutdown(self) -> None:
        if self.wandb_reporter:
            self.wandb_reporter.shutdown_reporter()

    def poll_clients_for_sample_counts(self, timeout: Optional[float]) -> List[int]:

        # Poll clients for sample counts
        log(INFO, "Polling Clients for sample counts")
        assert isinstance(self.strategy, FedAvgSampling)
        client_instructions = self.strategy.configure_poll(server_round=1, client_manager=self._client_manager)
        results, _ = poll_clients(
            client_instructions=client_instructions, max_workers=self.max_workers, timeout=timeout
        )

        sample_counts: List[int] = [
            int(get_properties_res.properties["num_samples"]) for (_, get_properties_res) in results
        ]

        return sample_counts
