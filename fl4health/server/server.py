from logging import INFO
from typing import Dict, List, Optional, Tuple

import numpy as np
from flwr.common import Scalar
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.history import History
from flwr.server.server import EvaluateResultsAndFailures, Server
from flwr.server.strategy import Strategy

from fl4health.checkpointing.checkpointer import TorchCheckpointer
from fl4health.reporting.fl_wanb import ServerWandBReporter
from fl4health.server.polling import poll_clients
from fl4health.strategies.client_dp_fedavgm import ClientLevelDPFedAvgM
from fl4health.strategies.fedavg_with_extra_variables import FedAvgWithExtraVariables


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


class FedProxServer(FlServer):
    """
    Server to be used in case of FedProx with adaptive proximal weight.
    Modified the evaluate function to evaluate the performance of server and increase proximal weight if loss goes up
    or reduce proxmial weight if loss goes down for proximal_weight_patience consecutive rounds.
    """

    def __init__(
        self,
        client_manager: ClientManager,
        strategy: FedAvgWithExtraVariables,
        wandb_reporter: Optional[ServerWandBReporter] = None,
        checkpointer: Optional[TorchCheckpointer] = None,
        adaptive_proximal_weight: bool = True,
    ) -> None:
        super().__init__(
            client_manager=client_manager, strategy=strategy, wandb_reporter=wandb_reporter, checkpointer=checkpointer
        )

        self.strategy: FedAvgWithExtraVariables = strategy
        self.adaptive_proximal_weight: bool = adaptive_proximal_weight
        self.proximal_weight = strategy.server_extra_variables["proximal_weight"][0]
        self.previous_loss = float("inf")

        if self.adaptive_proximal_weight:
            self.proximal_weight_patience_counter: int = 0
            self.proximal_weight_patience: int = 5
            self.proximal_weight_delta: float = 0.1

    def evaluate_round(
        self,
        server_round: int,
        timeout: Optional[float],
    ) -> Optional[Tuple[Optional[float], Dict[str, Scalar], EvaluateResultsAndFailures]]:
        """Validate current global model on a number of clients."""

        super_result = super().evaluate_round(server_round, timeout)

        # Update the proximal weight parameter if adaptive proximal weight is enabled and the loss is not None
        if super_result is not None:
            loss_aggregated, _, (_, _) = super_result

            if loss_aggregated is not None:
                self._maybe_update_proximal_weight_param(self.previous_loss, float(loss_aggregated))
                self.previous_loss = float(loss_aggregated)
            self.strategy.set_extra_variables(["proximal_weight"], [[np.array(self.proximal_weight)]])

        return super_result

    def _maybe_update_proximal_weight_param(self, previous_loss: float, loss: float) -> None:

        if self.adaptive_proximal_weight:
            if loss <= previous_loss:
                self.proximal_weight_patience_counter += 1
                if self.proximal_weight_patience_counter == self.proximal_weight_patience:
                    self.proximal_weight -= self.proximal_weight_delta
                    self.proximal_weight = np.maximum(0.0, self.proximal_weight)
                    self.proximal_weight_patience_counter = 0
                    log(INFO, f"Proximal weight is decrease to {self.proximal_weight}")
            else:
                self.proximal_weight += self.proximal_weight_delta
                log(INFO, f"Proximal weight is increased to {self.proximal_weight}")
        return None
