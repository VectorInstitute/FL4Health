from typing import Optional, Sequence, Union

from flwr.common.typing import Config
from flwr.server.client_manager import ClientManager

from fl4health.checkpointing.checkpointer import TorchCheckpointer
from fl4health.reporting.base_reporter import BaseReporter
from fl4health.servers.base_server import FlServer
from fl4health.strategies.fedavg_with_adaptive_constraint import FedAvgWithAdaptiveConstraint


class MrMtlServer(FlServer):
    def __init__(
        self,
        client_manager: ClientManager,
        fl_config: Config,
        strategy: FedAvgWithAdaptiveConstraint,
        checkpointer: Optional[Union[TorchCheckpointer, Sequence[TorchCheckpointer]]] = None,
        reporters: Sequence[BaseReporter] | None = None,
    ) -> None:
        """
        This is a very basic wrapper class over the FlServer to ensure that the strategy used for MR-MTL is of type
        FedAvgWithAdaptiveConstraint.

        Args:
            client_manager (ClientManager): Determines the mechanism by which clients are sampled by the server, if
                they are to be sampled at all.
            fl_config (Config): This should be the configuration that was used to setup the federated training.
                In most cases it should be the "source of truth" for how FL training/evaluation should proceed. For
                example, the config used to produce the on_fit_config_fn and on_evaluate_config_fn for the strategy.
                NOTE: This config is DISTINCT from the Flwr server config, which is extremely minimal.
            strategy (FedAvgWithAdaptiveConstraint): The aggregation strategy to be used by the server to handle.
                client updates and other information potentially sent by the participating clients. For MR-MTL, the
                strategy must be a derivative of the FedAvgWithAdaptiveConstraint class.
            checkpointer (Optional[Union[TorchCheckpointer, Sequence [TorchCheckpointer]]], optional): To be provided
                if the server should perform server side checkpointing based on some criteria. If none, then no
                server-side checkpointing is performed. Multiple checkpointers can also be passed in a sequence to
                checkpointer based on multiple criteria. Ensure checkpoint names are different for each checkpoint
                or they will overwrite on another. Defaults to None.
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
            checkpointer=checkpointer,
            reporters=reporters,
        )
