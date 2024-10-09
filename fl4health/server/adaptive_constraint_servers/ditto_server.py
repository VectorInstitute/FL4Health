from typing import Optional, Sequence, Union

from flwr.server.client_manager import ClientManager

from fl4health.checkpointing.checkpointer import TorchCheckpointer
from fl4health.reporting.fl_wandb import ServerWandBReporter
from fl4health.reporting.metrics import MetricsReporter
from fl4health.server.base_server import FlServer
from fl4health.strategies.fedavg_with_adaptive_constraint import FedAvgWithAdaptiveConstraint


class DittoServer(FlServer):
    def __init__(
        self,
        client_manager: ClientManager,
        strategy: FedAvgWithAdaptiveConstraint,
        wandb_reporter: Optional[ServerWandBReporter] = None,
        checkpointer: Optional[Union[TorchCheckpointer, Sequence[TorchCheckpointer]]] = None,
        metrics_reporter: Optional[MetricsReporter] = None,
    ) -> None:
        """
        This is a very basic wrapper class over the FlServer to ensure that the strategy used for Ditto is of type
        FedAvgWithAdaptiveConstraint.

        Args:
            client_manager (ClientManager): Determines the mechanism by which clients are sampled by the server, if
                they are to be sampled at all.
            strategy (FedAvgWithAdaptiveConstraint): The aggregation strategy to be used by the server to handle.
                client updates and other information potentially sent by the participating clients. For Ditto, the
                strategy must be a derivative of the FedAvgWithAdaptiveConstraint class.
            wandb_reporter (Optional[ServerWandBReporter], optional): To be provided if the server is to log
                information and results to a Weights and Biases account. If None is provided, no logging occurs.
                Defaults to None.
            checkpointer (Optional[Union[TorchCheckpointer, Sequence [TorchCheckpointer]]], optional): To be provided
                if the server should perform server side checkpointing based on some criteria. If none, then no
                server-side checkpointing is performed. Multiple checkpointers can also be passed in a sequence to
                checkpointer based on multiple criteria. Ensure checkpoint names are different for each checkpoint
                or they will overwrite on another. Defaults to None.
            metrics_reporter (Optional[MetricsReporter], optional): A metrics reporter instance to record the metrics
                during the execution. Defaults to an instance of MetricsReporter with default init parameters.
        """
        assert isinstance(
            strategy, FedAvgWithAdaptiveConstraint
        ), "Strategy must be of base type FedAvgWithAdaptiveConstraint"
        super().__init__(client_manager, strategy, wandb_reporter, checkpointer, metrics_reporter)
