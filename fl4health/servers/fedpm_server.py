from collections.abc import Callable, Sequence

from flwr.common import Parameters
from flwr.common.typing import Config, Scalar
from flwr.server.client_manager import ClientManager
from flwr.server.server import FitResultsAndFailures

from fl4health.checkpointing.server_module import LayerNamesServerCheckpointAndStateModule
from fl4health.reporting.base_reporter import BaseReporter
from fl4health.servers.base_server import FlServer
from fl4health.strategies.fedpm import FedPm


class FedPmServer(FlServer):
    def __init__(
        self,
        client_manager: ClientManager,
        fl_config: Config,
        strategy: FedPm,
        reporters: Sequence[BaseReporter] | None = None,
        checkpoint_and_state_module: LayerNamesServerCheckpointAndStateModule | None = None,
        on_init_parameters_config_fn: Callable[[int], dict[str, Scalar]] | None = None,
        server_name: str | None = None,
        accept_failures: bool = True,
        reset_frequency: int = 1,
    ) -> None:
        """
        Custom FL Server for the FedPM algorithm to allow for resetting the beta priors in Bayesian aggregation,
        as specified in http://arxiv.org/pdf/2209.15328.

        Args:
            client_manager (ClientManager): Determines the mechanism by which clients are sampled by the server, if
                they are to be sampled at all.
            fl_config (Config): This should be the configuration that was used to setup the federated training.
                In most cases it should be the "source of truth" for how FL training/evaluation should proceed. For
                example, the config used to produce the ``on_fit_config_fn`` and ``on_evaluate_config_fn`` for the
                strategy.

                **NOTE**: This config is **DISTINCT** from the Flwr server config, which is extremely minimal.
            strategy (FedPm): The aggregation strategy to be used by the server to handle client updates and other
                information potentially sent by the participating clients. This strategy must be of FedPm type.
            reporters (Sequence[BaseReporter] | None, optional): A sequence of FL4Health reporters which the server
                should send data to before and after each round.
            checkpoint_and_state_module (LayerNamesServerCheckpointAndStateModule | None, optional): This module is
                used to handle both model checkpointing and state checkpointing. The former is aimed at saving model
                artifacts to be used or evaluated after training. The latter is used to preserve training state
                (including models) such that if FL training is interrupted, the process may be restarted. If no
                module is provided, no checkpointing or state preservation will happen. Defaults to None.
            on_init_parameters_config_fn (Callable[[int], dict[str, Scalar]] | None, optional): Function used to
                configure how one asks a client to provide parameters from which to initialize all other clients by
                providing a ``Config`` dictionary. If this is none, then a blank config is sent with the parameter
                request (which is default behavior for flower servers). Defaults to None.
            server_name (str | None, optional): An optional string name to uniquely identify server. This name is also
                used as part of any state checkpointing done by the server. Defaults to None.
            accept_failures (bool, optional): Determines whether the server should accept failures during training or
                evaluation from clients or not. If set to False, this will cause the server to shutdown all clients
                and throw an exception. Defaults to True.
            reset_frequency (int, optional): Determines the frequency with which the beta priors are reset.
                Defaults to 1.
        """
        if checkpoint_and_state_module is not None:
            assert isinstance(
                checkpoint_and_state_module,
                LayerNamesServerCheckpointAndStateModule,
            ), "checkpoint_and_state_module must have type LayerNamesServerCheckpointAndStateModule"
        super().__init__(
            client_manager=client_manager,
            fl_config=fl_config,
            strategy=strategy,
            reporters=reporters,
            checkpoint_and_state_module=checkpoint_and_state_module,
            on_init_parameters_config_fn=on_init_parameters_config_fn,
            server_name=server_name,
            accept_failures=accept_failures,
        )
        self.reset_frequency = reset_frequency

    def fit_round(
        self,
        server_round: int,
        timeout: float | None,
    ) -> tuple[Parameters | None, dict[str, Scalar], FitResultsAndFailures] | None:
        assert isinstance(self.strategy, FedPm)
        # If self.reset_frequency == x, then the beta priors are reset every x fitting rounds.
        # Note that (server_round + 1) % self.reset_frequency == 0 is to ensure that the priors
        # are not reset in the second round when self.reset_frequency is 2.
        if server_round > 1 and (server_round + 1) % self.reset_frequency == 0:
            self.strategy.reset_beta_priors()
        return super().fit_round(server_round, timeout)
