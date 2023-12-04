from logging import INFO
from math import ceil
from typing import List, Optional

from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.history import History

from fl4health.checkpointing.checkpointer import TorchCheckpointer
from fl4health.client_managers.fixed_without_replacement_manager import FixedSamplingByFractionClientManager
from fl4health.client_managers.poisson_sampling_manager import PoissonSamplingClientManager
from fl4health.privacy.fl_accountants import (
    ClientLevelAccountant,
    FlClientLevelAccountantFixedSamplingNoReplacement,
    FlClientLevelAccountantPoissonSampling,
)
from fl4health.reporting.fl_wanb import ServerWandBReporter
from fl4health.server.base_server import FlServer
from fl4health.strategies.client_dp_fedavgm import ClientLevelDPFedAvgM


class ClientLevelDPFedAvgServer(FlServer):
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
        """
        Server to be used in case of Client Level Differential Privacy with Federated Averaging.

        Args:
            client_manager (ClientManager): Determines the mechanism by which clients are sampled by the server, if
                they are to be sampled at all.
            strategy (ClientLevelDPFedAvgM): The aggregation strategy to be used by the server to handle.
                client updates and other information potentially sent by the participating clients.
            server_noise_multiplier (float): Magnitude of noise added to the weights aggregation process by the server.
            num_server_rounds (int): Number of rounds of FL training carried out by the server.
            wandb_reporter (Optional[ServerWandBReporter], optional): To be provided if the server is to log
                information and results to a Weights and Biases account. If None is provided, no logging occurs.
                Defaults to None.
            checkpointer (Optional[TorchCheckpointer], optional): To be provided if the server should perform
                server side checkpointing based on some criteria. If none, then no server-side checkpointing is
                performed. Defaults to None.
            delta (Optional[float], optional): The delta value for epislon-delta DP accounting. If None it defaults to
                being 1/total_samples in the FL run. Defaults to None.
        """
        super().__init__(
            client_manager=client_manager, strategy=strategy, wandb_reporter=wandb_reporter, checkpointer=checkpointer
        )
        self.accountant: ClientLevelAccountant
        self.server_noise_multiplier = server_noise_multiplier
        self.num_server_rounds = num_server_rounds
        self.delta = delta

    def fit(self, num_rounds: int, timeout: Optional[float]) -> History:
        """
        Run federated averaging for a number of rounds.

        Args:
            num_rounds (int): Number of server rounds to run.
            timeout (Optional[float]): The amount of time in seconds that the server will wait for results from the
                clients selected to participate in federated training.

        Returns:
            History: The history object contains the full set of FL training results, including things like aggregated
                loss and metrics.
        """

        assert isinstance(self.strategy, ClientLevelDPFedAvgM)

        sample_counts = self.poll_clients_for_sample_counts(timeout)

        # If Weighted FedAvg, set sample counts to compute client weights
        if self.strategy.weighted_aggregation:
            self.strategy.sample_counts = sample_counts

        self.setup_privacy_accountant(sample_counts)

        return super().fit(num_rounds=num_rounds, timeout=timeout)

    def setup_privacy_accountant(self, sample_counts: List[int]) -> None:
        """
        Sets up FL Accountant and computes privacy loss based on class attributes and retrieved sample counts.

        Args:
            sample_counts (List[int]): These should be the total number of training examples fetched from all clients
                during the sample polling process.
        """
        assert isinstance(self.strategy, ClientLevelDPFedAvgM)

        num_clients = len(sample_counts)
        target_delta = self.delta if self.delta is not None else 1 / num_clients

        if isinstance(self._client_manager, PoissonSamplingClientManager):
            self.accountant = FlClientLevelAccountantPoissonSampling(
                client_sampling_rate=self.strategy.fraction_fit, noise_multiplier=self.server_noise_multiplier
            )
        else:
            assert isinstance(self._client_manager, FixedSamplingByFractionClientManager)
            num_clients_sampled = ceil(len(sample_counts) * self.strategy.fraction_fit)
            self.accountant = FlClientLevelAccountantFixedSamplingNoReplacement(
                n_total_clients=num_clients,
                n_clients_sampled=num_clients_sampled,
                noise_multiplier=self.server_noise_multiplier,
            )

        # Note that this assumes that the FL round has exactly n_clients participating.
        epsilon = self.accountant.get_epsilon(self.num_server_rounds, target_delta)
        log(INFO, f"Model privacy after full training will be ({epsilon}, {target_delta})")
