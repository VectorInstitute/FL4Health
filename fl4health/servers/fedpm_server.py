from collections.abc import Sequence
from typing import Dict, Optional, Tuple

from flwr.common import Parameters
from flwr.common.typing import Config, Scalar
from flwr.server.client_manager import ClientManager
from flwr.server.server import FitResultsAndFailures

from fl4health.checkpointing.checkpointer import TorchCheckpointer
from fl4health.reporting.base_reporter import BaseReporter
from fl4health.servers.base_server import FlServer
from fl4health.strategies.fedpm import FedPm


class FedPmServer(FlServer):
    def __init__(
        self,
        client_manager: ClientManager,
        fl_config: Config,
        strategy: FedPm,
        checkpointer: Optional[TorchCheckpointer] = None,
        reset_frequency: int = 1,
        reporters: Sequence[BaseReporter] | None = None,
        accept_failures: bool = True,
    ) -> None:
        """
        Custom FL Server for the FedPM algorithm to allow for resetting the beta priors in Bayesian aggregation,
        as specified in http://arxiv.org/pdf/2209.15328.

        Args:
            client_manager (ClientManager): Determines the mechanism by which clients are sampled by the server, if
                they are to be sampled at all.
            fl_config (Config): This should be the configuration that was used to setup the federated training.
                In most cases it should be the "source of truth" for how FL training/evaluation should proceed. For
                example, the config used to produce the on_fit_config_fn and on_evaluate_config_fn for the strategy.
                NOTE: This config is DISTINCT from the Flwr server config, which is extremely minimal.
            strategy (Scaffold): The aggregation strategy to be used by the server to handle client updates and other
                information potentially sent by the participating clients. This strategy must be of SCAFFOLD type.
            checkpointer (Optional[TorchCheckpointer], optional): To be provided if the server should perform
                server side checkpointing based on some criteria. If none, then no server-side checkpointing is
                performed. Defaults to None.
            reset_frequency (int): Determines the frequency with which the beta priors are reset. Defaults to 1.
            reporters (Sequence[BaseReporter], optional): A sequence of FL4Health reporters which the server should
                send data to before and after each round.
            accept_failures (bool, optional): Determines whether the server should accept failures during training or
                evaluation from clients or not. If set to False, this will cause the server to shutdown all clients
                and throw an exception. Defaults to True.
        """
        FlServer.__init__(
            self,
            client_manager=client_manager,
            fl_config=fl_config,
            strategy=strategy,
            checkpointer=checkpointer,
            reporters=reporters,
            accept_failures=accept_failures,
        )
        self.reset_frequency = reset_frequency

    def fit_round(
        self,
        server_round: int,
        timeout: Optional[float],
    ) -> Optional[Tuple[Optional[Parameters], Dict[str, Scalar], FitResultsAndFailures]]:
        assert isinstance(self.strategy, FedPm)
        # If self.reset_frequency == x, then the beta priors are reset every x fitting rounds.
        # Note that (server_round + 1) % self.reset_frequency == 0 is to ensure that the priors
        # are not reset in the second round when self.reset_frequency is 2.
        if server_round > 1 and (server_round + 1) % self.reset_frequency == 0:
            self.strategy.reset_beta_priors()
        return super().fit_round(server_round, timeout)
