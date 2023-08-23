from logging import INFO
from typing import List, Optional

from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.history import History

from fl4health.checkpointing.checkpointer import TorchCheckpointer
from fl4health.privacy.fl_accountants import FlClientLevelAccountantPoissonSampling
from fl4health.reporting.fl_wanb import ServerWandBReporter
from fl4health.server.base_server import FlServer
from fl4health.strategies.client_dp_fedavgm import ClientLevelDPFedAvgM


class ClientLevelDPWeightedFedAvgServer(FlServer):
    """
    Server to be used in case of Client Level Differential Privacy with weighted Federated Averaging.
    Modified the fit function to poll clients for sample counts prior to the first round of FL.
    """

    def __init__(
        self,
        client_manager: ClientManager,
        strategy: ClientLevelDPFedAvgM,
        server_noise_multiplier: float,
        num_server_rounds: int,
        wandb_reporter: Optional[ServerWandBReporter] = None,
        checkpointer: Optional[TorchCheckpointer] = None,
        delta: Optional[int] = None,
    ) -> None:
        super().__init__(
            client_manager=client_manager, strategy=strategy, wandb_reporter=wandb_reporter, checkpointer=checkpointer
        )
        self.server_noise_multiplier = server_noise_multiplier
        self.num_server_rounds = num_server_rounds
        self.delta = delta

    def fit(self, num_rounds: int, timeout: Optional[float]) -> History:
        """Run federated averaging for a number of rounds."""

        assert isinstance(self.strategy, ClientLevelDPFedAvgM)

        sample_counts = self.poll_clients_for_sample_counts(timeout)

        # If Weighted FedAvg, set sample counts to compute client weights
        if self.strategy.weighted_averaging:
            self.strategy.sample_counts = sample_counts

        self.setup_privacy_accountant(sample_counts)

        return super().fit(num_rounds=num_rounds, timeout=timeout)

    def setup_privacy_accountant(self, sample_counts: List[int]) -> None:
        assert isinstance(self.strategy, ClientLevelDPFedAvgM)

        self.accountant = FlClientLevelAccountantPoissonSampling(
            client_sampling_rate=self.strategy.fraction_fit, noise_multiplier=self.server_noise_multiplier
        )

        num_clients = len(sample_counts)
        print(num_clients, "heyeeyeyeyey")
        target_delta = self.delta if self.delta is not None else 1 / num_clients

        # Note that this assumes that the FL round has exactly n_clients participating.
        epsilon = self.accountant.get_epsilon(self.num_server_rounds, target_delta)
        log(INFO, f"Model privacy after full training will be ({epsilon}, {target_delta})")
