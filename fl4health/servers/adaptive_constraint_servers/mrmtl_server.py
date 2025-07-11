from collections.abc import Callable, Sequence

from flwr.common.typing import Config, Scalar
from flwr.server.client_manager import ClientManager

from fl4health.checkpointing.server_module import AdaptiveConstraintServerCheckpointAndStateModule
from fl4health.reporting.base_reporter import BaseReporter
from fl4health.servers.base_server import FlServer
from fl4health.strategies.fedavg_with_adaptive_constraint import FedAvgWithAdaptiveConstraint


class MrMtlServer(FlServer):
    def __init__(
        self,
        client_manager: ClientManager,
        fl_config: Config,
        strategy: FedAvgWithAdaptiveConstraint,
        reporters: Sequence[BaseReporter] | None = None,
        checkpoint_and_state_module: AdaptiveConstraintServerCheckpointAndStateModule | None = None,
        on_init_parameters_config_fn: Callable[[int], dict[str, Scalar]] | None = None,
        server_name: str | None = None,
        accept_failures: bool = True,
    ) -> None:
        """
        This is a very basic wrapper class over the ``FlServer`` to ensure that the strategy used for MR-MTL is of type
        ``FedAvgWithAdaptiveConstraint``.

        Args:
            client_manager (ClientManager): Determines the mechanism by which clients are sampled by the server, if
                they are to be sampled at all.
            fl_config (Config): This should be the configuration that was used to setup the federated training.
                In most cases it should be the "source of truth" for how FL training/evaluation should proceed. For
                example, the config used to produce the ``on_fit_config_fn`` and ``on_evaluate_config_fn`` for the
                strategy.

                **NOTE**: This config is **DISTINCT** from the Flwr server config, which is extremely minimal.
            strategy (FedAvgWithAdaptiveConstraint): The aggregation strategy to be used by the server to handle.
                client updates and other information potentially sent by the participating clients. For MR-MTL, the
                strategy must be a derivative of the ``FedAvgWithAdaptiveConstraint`` class.
            reporters (Sequence[BaseReporter], optional): A sequence of FL4Health reporters which the server should
                send data to before and after each round. Defaults to None.
            checkpoint_and_state_module (AdaptiveConstraintServerCheckpointAndStateModule | None, optional): This
                module is used to handle both model checkpointing and state checkpointing. The former is aimed at
                saving model artifacts to be used or evaluated after training. The latter is used to preserve training
                state (including models) such that if FL training is interrupted, the process may be restarted. If no
                module is provided, no checkpointing or state preservation will happen. Defaults to None.

                **NOTE**: For MR-MTL, the server model is an aggregation of the personal models, which isn't the
                target of FL training for this algorithm. However, one may still want to save this model for other
                purposes.
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
        assert isinstance(strategy, FedAvgWithAdaptiveConstraint), (
            "Strategy must be of base type FedAvgWithAdaptiveConstraint"
        )
        if checkpoint_and_state_module is not None:
            assert isinstance(
                checkpoint_and_state_module,
                AdaptiveConstraintServerCheckpointAndStateModule,
            ), "checkpoint_and_state_module must have type AdaptiveConstraintServerCheckpointAndStateModule"
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
