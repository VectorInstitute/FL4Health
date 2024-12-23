from collections.abc import Sequence
from logging import INFO
from math import ceil
from typing import List, Optional, Tuple

import torch.nn as nn
from flwr.common.logger import log
from flwr.common.typing import Config
from flwr.server.client_manager import ClientManager
from flwr.server.history import History

from fl4health.checkpointing.opacus_checkpointer import OpacusCheckpointer
from fl4health.client_managers.poisson_sampling_manager import PoissonSamplingClientManager
from fl4health.privacy.fl_accountants import FlInstanceLevelAccountant
from fl4health.reporting.base_reporter import BaseReporter
from fl4health.servers.base_server import ExchangerType, FlServer
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
        local_epochs: Optional[int] = None,
        local_steps: Optional[int] = None,
        model: nn.Module | None = None,
        checkpointer: Optional[OpacusCheckpointer] = None,
        parameter_exchanger: ExchangerType | None = None,
        reporters: Sequence[BaseReporter] | None = None,
        delta: Optional[float] = None,
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
                example, the config used to produce the on_fit_config_fn and on_evaluate_config_fn for the strategy.
                NOTE: This config is DISTINCT from the Flwr server config, which is extremely minimal.
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
            strategy (OpacusBasicFedAvg): The aggregation strategy to be used by the server to handle
                client updates and other information potentially sent by the participating clients. this must be an
                OpacusBasicFedAvg strategy to ensure proper treatment of the model in the Opacus framework
            model (Optional[nn.Module]): This is the torch model to be checkpointed. It will be hydrated by the
                _hydrate_model_for_checkpointing function so that it has the proper weights to be saved. If no model
                is defined and checkpointing is attempted an error will throw. Defaults to None.
            checkpointer (Optional[OpacusCheckpointer], optional): To be provided if the server should perform
                server side checkpointing based on some criteria. If none, then no server-side checkpointing is
                performed. Defaults to None.
            parameter_exchanger (Optional[ExchangerType], optional): A parameter exchanger used to facilitate
                server-side model checkpointing if a checkpointer has been defined. If not provided then checkpointing
                will not be done unless the _hydrate_model_for_checkpointing function is overridden. Because the
                server only sees numpy arrays, the parameter exchanger is used to insert the numpy arrays into a
                provided model. Defaults to None.
            reporters (Sequence[BaseReporter], optional): A sequence of FL4Health
                reporters which the client should send data to.
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
            model=model,
            checkpointer=checkpointer,
            parameter_exchanger=parameter_exchanger,
            reporters=reporters,
            accept_failures=accept_failures,
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

    def fit(self, num_rounds: int, timeout: Optional[float]) -> Tuple[History, float]:
        """
        Run federated averaging for a number of rounds.

        Args:
            num_rounds (int): Number of server rounds to run.
            timeout (Optional[float]): The amount of time in seconds that the server will wait for results from the
                clients selected to participate in federated training.

        Returns:
            Tuple[History, float]: The first element of the tuple is a history object containing the full
                set of FL training results, including things like aggregated loss and metrics.
                Tuple also includes elapsed time in seconds for round.
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
        log(
            INFO,
            f"Model privacy after full training will be ({epsilon}, {target_delta})",
        )
