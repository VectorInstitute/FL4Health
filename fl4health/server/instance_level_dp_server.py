from logging import INFO
from math import ceil
from typing import List, Optional

from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.history import History

from fl4health.checkpointing.checkpointer import TorchCheckpointer
from fl4health.client_managers.poisson_sampling_manager import PoissonSamplingClientManager
from fl4health.privacy.fl_accountants import FlInstanceLevelAccountant
from fl4health.reporting.fl_wanb import ServerWandBReporter
from fl4health.server.base_server import FlServer
from fl4health.strategies.basic_fedavg import BasicFedAvg
from fl4health.strategies.strategy_with_poll import StrategyWithPolling


class InstanceLevelDPServer(FlServer):
    def __init__(
        self,
        client_manager: ClientManager,
        noise_multiplier: float,
        batch_size: int,
        num_server_rounds: int,
        strategy: BasicFedAvg,
        local_epochs: Optional[int] = None,
        local_steps: Optional[int] = None,
        wandb_reporter: Optional[ServerWandBReporter] = None,
        checkpointer: Optional[TorchCheckpointer] = None,
        delta: Optional[float] = None,
    ) -> None:
        """
        Server to be used in case of Instance Level Differential Privacy with Federated Averaging.
        Modified the fit function to poll clients for sample counts prior to the first round of FL.

        Args:
            client_manager (ClientManager): Determines the mechanism by which clients are sampled by the server, if
                they are to be sampled at all.
            noise_multiplier (int): The amount of Gaussian noise to be added to the per sample gradient during
                DP-SGD.
            batch_size (int): The batch size to be used in training on the client-side. Used in privacy accounting.
            num_server_rounds (int): The number of server rounds to be done in FL training. Used in privacy accounting
            local_epochs (Optional[int], optional): Number of local epochs to be performed on the client-side. This is
                used in privacy accounting. One of local_epochs or local_steps should be defined, but not both.
                Defaults to None.
            local_steps (Optional[int], optional): Number of local steps to be performed on the client-side. This is
                used in privacy accounting. One of local_epochs or local_steps should be defined, but not both.
                Defaults to None.
            strategy (BasicFedAvg): The aggregation strategy to be used by the server to handle
                client updates and other information potentially sent by the participating clients.
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
            client_manager=client_manager,
            strategy=strategy,
            wandb_reporter=wandb_reporter,
            checkpointer=checkpointer,
        )

        # Ensure that one of local_epochs and local_steps is passed (and not both)
        assert isinstance(local_epochs, int) ^ isinstance(local_steps, int)
        self.accountant: FlInstanceLevelAccountant
        self.local_steps = local_steps
        self.local_epochs = local_epochs

        # Whether or not we have to convert local_steps to local_epochs
        self.convert_steps_to_epochs = True if self.local_epochs is None else False
        self.noise_multiplier = noise_multiplier
        self.batch_size = batch_size
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

        assert isinstance(self.strategy, StrategyWithPolling)
        sample_counts = self.poll_clients_for_sample_counts(timeout)
        self.setup_privacy_accountant(sample_counts)

        return super().fit(num_rounds=num_rounds, timeout=timeout)

    def setup_privacy_accountant(self, sample_counts: List[int]) -> None:
        """
        Sets up FL Accountant and computes privacy loss based on class attributes and retrieved sample counts.

        Args:
            sample_counts (List[int]): These should be the total number of training examples fetched from all clients
                during the sample polling process.
        """
        # Ensures that we're using a fraction sampler of the
        assert isinstance(self._client_manager, PoissonSamplingClientManager)

        total_samples = sum(sample_counts)

        if self.convert_steps_to_epochs:
            # Compute the ceiling of number of local epochs per clients and take max as local epochs
            # Ensures we do not underestimate the privacy loss
            assert isinstance(self.local_steps, int)

            epochs_per_client = [ceil(self.local_steps * self.batch_size / count) for count in sample_counts]
            self.local_epochs = max(epochs_per_client)

        assert isinstance(self.local_epochs, int)
        assert isinstance(self.strategy, BasicFedAvg)

        self.accountant = FlInstanceLevelAccountant(
            client_sampling_rate=self.strategy.fraction_fit,
            noise_multiplier=self.noise_multiplier,
            epochs_per_round=self.local_epochs,
            client_batch_sizes=[self.batch_size] * len(sample_counts),
            client_dataset_sizes=sample_counts,
        )

        target_delta = 1.0 / total_samples if self.delta is None else self.delta
        epsilon = self.accountant.get_epsilon(self.num_server_rounds, target_delta)
        log(INFO, f"Model privacy after full training will be ({epsilon}, {target_delta})")
