from collections.abc import Sequence
from logging import INFO
from math import ceil
from typing import List, Optional, Tuple

from flwr.common.logger import log
from flwr.common.typing import Config
from flwr.server.client_manager import ClientManager
from flwr.server.history import History

from fl4health.checkpointing.server_module import BaseServerCheckpointAndStateModule
from fl4health.client_managers.fixed_without_replacement_manager import FixedSamplingByFractionClientManager
from fl4health.client_managers.poisson_sampling_manager import PoissonSamplingClientManager
from fl4health.privacy.fl_accountants import (
    ClientLevelAccountant,
    FlClientLevelAccountantFixedSamplingNoReplacement,
    FlClientLevelAccountantPoissonSampling,
)
from fl4health.reporting.base_reporter import BaseReporter
from fl4health.servers.base_server import FlServer
from fl4health.strategies.client_dp_fedavgm import ClientLevelDPFedAvgM


class ClientLevelDPFedAvgServer(FlServer):
    def __init__(
        self,
        client_manager: ClientManager,
        fl_config: Config,
        strategy: ClientLevelDPFedAvgM,
        server_noise_multiplier: float,
        num_server_rounds: int,
        checkpoint_and_state_module: BaseServerCheckpointAndStateModule | None = None,
        reporters: Sequence[BaseReporter] | None = None,
        delta: Optional[int] = None,
        accept_failures: bool = True,
    ) -> None:
        """
        Server to be used in case of Client Level Differential Privacy with Federated Averaging.

        Args:
            client_manager (ClientManager): Determines the mechanism by which clients are sampled by the server, if
                they are to be sampled at all.
            fl_config (Config): This should be the configuration that was used to setup the federated training.
                In most cases it should be the "source of truth" for how FL training/evaluation should proceed. For
                example, the config used to produce the on_fit_config_fn and on_evaluate_config_fn for the strategy.
                NOTE: This config is DISTINCT from the Flwr server config, which is extremely minimal.
            strategy (ClientLevelDPFedAvgM): The aggregation strategy to be used by the server to handle.
                client updates and other information potentially sent by the participating clients.
            server_noise_multiplier (float): Magnitude of noise added to the weights aggregation process by the server.
            num_server_rounds (int): Number of rounds of FL training carried out by the server.
            checkpoint_and_state_module (BaseServerCheckpointAndStateModule | None, optional): This module is used
                to handle both model checkpointing and state checkpointing. The former is aimed at saving model
                artifacts to be used or evaluated after training. The later is used to preserve training state
                (including models) such that if FL training is interrupted, the process may be restarted. If no
                module is provided, no checkpointing or state preservation will happen. Defaults to None.
            reporters (Sequence[BaseReporter], optional): A sequence of FL4Health
                reporters which the server should send data to before and after each round.
            delta (Optional[float], optional): The delta value for epsilon-delta DP accounting. If None it defaults to
                being 1/total_samples in the FL run. Defaults to None.
            accept_failures (bool, optional): Determines whether the server should accept failures during training or
                evaluation from clients or not. If set to False, this will cause the server to shutdown all clients
                and throw an exception. Defaults to True.
        """
        super().__init__(
            client_manager=client_manager,
            fl_config=fl_config,
            strategy=strategy,
            checkpoint_and_state_module=checkpoint_and_state_module,
            reporters=reporters,
            accept_failures=accept_failures,
        )
        self.accountant: ClientLevelAccountant
        self.server_noise_multiplier = server_noise_multiplier
        self.num_server_rounds = num_server_rounds
        self.delta = delta

    def fit(self, num_rounds: int, timeout: Optional[float]) -> Tuple[History, float]:
        """
        Run federated averaging for a number of rounds.

        Args:
            num_rounds (int): Number of server rounds to run.
            timeout (Optional[float]): The amount of time in seconds that the server will wait for results from the
                clients selected to participate in federated training.

        Returns:
            Tuple[History, float]: The first element of the tuple is a history object containing the full set of
                FL training results, including things like aggregated loss and metrics.
                Tuple also contains the elapsed time in seconds for the round.
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
                client_sampling_rate=self.strategy.fraction_fit,
                noise_multiplier=self.server_noise_multiplier,
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
        log(
            INFO,
            f"Model privacy after full training will be ({epsilon}, {target_delta})",
        )
