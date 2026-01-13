from collections.abc import Callable, Sequence
from logging import INFO
from math import ceil

from flwr.common.logger import log
from flwr.common.typing import Config, Scalar
from flwr.server.client_manager import ClientManager
from flwr.server.history import History

from fl4health.checkpointing.server_module import OpacusServerCheckpointAndStateModule
from fl4health.client_managers.poisson_sampling_manager import PoissonSamplingClientManager
from fl4health.privacy.fl_accountants import FlInstanceLevelAccountant
from fl4health.reporting.base_reporter import BaseReporter
from fl4health.servers.base_server import FlServer
from fl4health.strategies.basic_fedavg import BasicFedAvg
from fl4health.strategies.strategy_with_poll import StrategyWithPolling


class InstanceLevelDpServer(FlServer):
    def __init__(
        self,
        client_manager: ClientManager,
        fl_config: Config,
        noise_multiplier: float,
        batch_size: int,
        num_server_rounds: int,
        strategy: BasicFedAvg,
        local_epochs: int | None = None,
        local_steps: int | None = None,
        checkpoint_and_state_module: OpacusServerCheckpointAndStateModule | None = None,
        reporters: Sequence[BaseReporter] | None = None,
        delta: float | None = None,
        on_init_parameters_config_fn: Callable[[int], dict[str, Scalar]] | None = None,
        server_name: str | None = None,
        accept_failures: bool = True,
    ) -> None:
        """
        Server to be used in case of Instance Level Differential Privacy with Federated Averaging.
        Modified the fit function to poll clients for sample counts prior to the first round of FL.

        Args:
            client_manager (ClientManager): Determines the mechanism by which clients are sampled by the server, if
                they are to be sampled at all.
            fl_config (Config): This should be the configuration that was used to setup the federated training.
                In most cases it should be the "source of truth" for how FL training/evaluation should proceed. For
                example, the config used to produce the ``on_fit_config_fn`` and ``on_evaluate_config_fn`` for the
                strategy.

                **NOTE**: This config is **DISTINCT** from the Flwr server config, which is extremely minimal.
            noise_multiplier (float): The amount of Gaussian noise to be added to the per sample gradient during
                DP-SGD.
            batch_size (int): The batch size to be used in training on the client-side. Used in privacy accounting.
            num_server_rounds (int): The number of server rounds to be done in FL training. Used in privacy accounting
            strategy (BasicFedAvg): The aggregation strategy to be used by the server to handle
                client updates and other information potentially sent by the participating clients. this must be an
                ``OpacusBasicFedAvg`` strategy to ensure proper treatment of the model in the Opacus framework
            local_epochs (int | None, optional): Number of local epochs to be performed on the client-side. This is
                used in privacy accounting. One of ``local_epochs`` or ``local_steps`` should be defined, but not both.
                Defaults to None.
            local_steps (int | None, optional): Number of local steps to be performed on the client-side. This is
                used in privacy accounting. One of ``local_epochs`` or ``local_steps`` should be defined, but not both.
                Defaults to None.
            checkpoint_and_state_module (OpacusServerCheckpointAndStateModule | None, optional): This module is used
                to handle both model checkpointing and state checkpointing. The former is aimed at saving model
                artifacts to be used or evaluated after training. The latter is used to preserve training state
                (including models) such that if FL training is interrupted, the process may be restarted. If no
                module is provided, no checkpointing or state preservation will happen. Defaults to None.
            reporters (Sequence[BaseReporter] | None, optional): A sequence of FL4Health reporters which the client
                should send data to.
            delta (float | None, optional): The delta value for epsilon-delta DP accounting. If None it defaults to
                being ``1/total_samples`` in the FL run. Defaults to None.
            on_init_parameters_config_fn (Callable[[int], dict[str, Scalar]] | None, optional): Function used to
                configure how one asks a client to provide parameters from which to initialize all other clients by
                providing a ``Config`` dictionary. If this is none, then a blank config is sent with the parameter
                request (which is default behavior for flower servers). Defaults to None.
            server_name (str | None, optional): An optional string name to uniquely identify server. This name is also
                used as part of any state checkpointing done by the server. Defaults to None.
            accept_failures (bool, optional): Determines whether the server should accept failures during training or
                evaluation from clients or not. If set to False, this will cause the server to shutdown all clients
                and throw an exception. Defaults to True.
        """
        if checkpoint_and_state_module is not None:
            assert isinstance(
                checkpoint_and_state_module,
                OpacusServerCheckpointAndStateModule,
            ), "checkpoint_and_state_module must have type OpacusServerCheckpointAndStateModule"
        super().__init__(
            client_manager=client_manager,
            fl_config=fl_config,
            strategy=strategy,
            checkpoint_and_state_module=checkpoint_and_state_module,
            reporters=reporters,
            on_init_parameters_config_fn=on_init_parameters_config_fn,
            server_name=server_name,
            accept_failures=accept_failures,
        )

        # Ensure that one of local_epochs and local_steps is passed (and not both)
        assert isinstance(local_epochs, int) ^ isinstance(local_steps, int)
        self.accountant: FlInstanceLevelAccountant
        self.local_steps = local_steps
        self.local_epochs = local_epochs

        # Whether or not we have to convert local_steps to local_epochs
        self.convert_steps_to_epochs = self.local_epochs is None
        self.noise_multiplier = noise_multiplier
        self.batch_size = batch_size
        self.num_server_rounds = num_server_rounds
        self.delta = delta

    def fit(self, num_rounds: int, timeout: float | None) -> tuple[History, float]:
        """
        Run federated averaging for a number of rounds.

        Args:
            num_rounds (int): Number of server rounds to run.
            timeout (float | None): The amount of time in seconds that the server will wait for results from the
                clients selected to participate in federated training.

        Returns:
            (tuple[History, float]): The first element of the tuple is a ``History`` object containing the full set of
                FL training results, including things like aggregated loss and metrics. Tuple also includes elapsed
                time in seconds for round.
        """
        assert isinstance(self.strategy, StrategyWithPolling)
        sample_counts = self.poll_clients_for_sample_counts(timeout)
        self.setup_privacy_accountant(sample_counts)

        return super().fit(num_rounds=num_rounds, timeout=timeout)

    def setup_privacy_accountant(self, sample_counts: list[int]) -> None:
        """
        Sets up FL Accountant and computes privacy loss based on class attributes and retrieved sample counts.

        Args:
            sample_counts (list[int]): These should be the total number of training examples fetched from all clients
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
        log(
            INFO,
            f"Model privacy after full training will be ({epsilon}, {target_delta})",
        )
