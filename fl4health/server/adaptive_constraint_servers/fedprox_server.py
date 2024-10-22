from typing import Optional, Sequence, Union

import torch.nn as nn
from flwr.common.parameter import parameters_to_ndarrays
from flwr.server.client_manager import ClientManager

from fl4health.checkpointing.checkpointer import TorchCheckpointer
from fl4health.parameter_exchange.packing_exchanger import FullParameterExchangerWithPacking
from fl4health.parameter_exchange.parameter_packer import ParameterPackerAdaptiveConstraint
from fl4health.reporting.base_reporter import BaseReporter
from fl4health.server.base_server import FlServerWithCheckpointing
from fl4health.strategies.fedavg_with_adaptive_constraint import FedAvgWithAdaptiveConstraint


class FedProxServer(FlServerWithCheckpointing[FullParameterExchangerWithPacking]):
    def __init__(
        self,
        client_manager: ClientManager,
        strategy: FedAvgWithAdaptiveConstraint,
        model: Optional[nn.Module] = None,
        checkpointer: Optional[Union[TorchCheckpointer, Sequence[TorchCheckpointer]]] = None,
        reporters: Sequence[BaseReporter] | None = None,
    ) -> None:
        """
        This is a wrapper class around FlServerWithCheckpointing for using the FedProx method that enforces that the
        parameter exchanger is a FullParameterExchangerWithPacking of the right type for model rehydration and that
        the strategy is of type FedAvgWithAdaptiveConstraint.

        Args:
            client_manager (ClientManager): Determines the mechanism by which clients are sampled by the server, if
                they are to be sampled at all.
            parameter_exchanger (ExchangerType): This is the parameter exchanger to be used to hydrate the model.
            strategy (FedAvgWithAdaptiveConstraint): The aggregation strategy to be used
                by the server to handle. client updates and other information
                potentially sent by the participating clients. For FedProx, the strategy
                must be a derivative of the FedAvgWithAdaptiveConstraint class.
            model (Optional[nn.Module], optional): This is the torch model to be
                hydrated by the _hydrate_model_for_checkpointing function, Defaults to
                None
            checkpointer (Optional[Union[TorchCheckpointer, Sequence[TorchCheckpointer]]], optional): To be provided
                if the server should perform server side checkpointing based on some
                criteria. If none, then no server-side checkpointing is performed.
                Multiple checkpointers can also be passed in a sequence to checkpoint
                based on multiple criteria. Defaults to None.
            reporters (Sequence[BaseReporter], optional): A sequence of FL4Health
                reporters which the server should send data to before and after each round.
        """
        assert isinstance(
            strategy, FedAvgWithAdaptiveConstraint
        ), "Strategy must be of base type FedAvgWithAdaptiveConstraint"
        parameter_exchanger = FullParameterExchangerWithPacking(ParameterPackerAdaptiveConstraint())
        super().__init__(
            client_manager=client_manager,
            parameter_exchanger=parameter_exchanger,
            model=model,
            strategy=strategy,
            checkpointer=checkpointer,
            reporters=reporters,
        )

    def _hydrate_model_for_checkpointing(self) -> nn.Module:
        assert (
            self.server_model is not None
        ), "Model hydration has been called but no server_model is defined to hydrate"
        # Overriding the standard hydration method to account for the unpacking
        packed_parameters = parameters_to_ndarrays(self.parameters)
        # Don't need the extra loss weight variable for checkpointing.
        model_ndarrays, _ = self.parameter_exchanger.unpack_parameters(packed_parameters)
        self.parameter_exchanger.pull_parameters(model_ndarrays, self.server_model)
        return self.server_model
