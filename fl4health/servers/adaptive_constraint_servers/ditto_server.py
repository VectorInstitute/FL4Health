from typing import Sequence

from flwr.common.typing import Config
from flwr.server.client_manager import ClientManager

from fl4health.checkpointing.server_module import BaseServerCheckpointAndStateModule
from fl4health.reporting.base_reporter import BaseReporter
from fl4health.servers.base_server import FlServer
from fl4health.strategies.fedavg_with_adaptive_constraint import FedAvgWithAdaptiveConstraint


class DittoServer(FlServer):
    def __init__(
        self,
        client_manager: ClientManager,
        fl_config: Config,
        strategy: FedAvgWithAdaptiveConstraint,
        checkpoint_and_state_module: BaseServerCheckpointAndStateModule | None = None,
        reporters: Sequence[BaseReporter] | None = None,
    ) -> None:
        """
        This is a very basic wrapper class over the FlServer to ensure that the strategy used for Ditto is of type
        FedAvgWithAdaptiveConstraint.

        Args:
            client_manager (ClientManager): Determines the mechanism by which clients are sampled by the server, if
                they are to be sampled at all.
            fl_config (Config): This should be the configuration that was used to setup the federated training.
                In most cases it should be the "source of truth" for how FL training/evaluation should proceed. For
                example, the config used to produce the on_fit_config_fn and on_evaluate_config_fn for the strategy.
                NOTE: This config is DISTINCT from the Flwr server config, which is extremely minimal.
            strategy (FedAvgWithAdaptiveConstraint): The aggregation strategy to be used by the server to handle.
                client updates and other information potentially sent by the participating clients. For Ditto, the
                strategy must be a derivative of the FedAvgWithAdaptiveConstraint class.
            checkpoint_and_state_module (BaseServerCheckpointAndStateModule | None, optional): This module is used
                to handle both model checkpointing and state checkpointing. The former is aimed at saving model
                artifacts to be used or evaluated after training. The later is used to preserve training state
                (including models) such that if FL training is interrupted, the process may be restarted. If no
                module is provided, no checkpointing or state preservation will happen. Defaults to None.
                NOTE: For Ditto, the model shared with the server is the GLOBAL MODEL, which isn't the target of FL
                training for this algorithm. However, one may still want to save this model for other purposes.
            reporters (Sequence[BaseReporter], optional): A sequence of FL4Health
                reporters which the server should send data to before and after each round.
        """
        assert isinstance(
            strategy, FedAvgWithAdaptiveConstraint
        ), "Strategy must be of base type FedAvgWithAdaptiveConstraint"
        super().__init__(
            client_manager=client_manager,
            fl_config=fl_config,
            strategy=strategy,
            checkpoint_and_state_module=checkpoint_and_state_module,
            reporters=reporters,
        )
