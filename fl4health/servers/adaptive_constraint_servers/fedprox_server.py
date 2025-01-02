from typing import Optional, Sequence, Union

import torch.nn as nn
from flwr.common.parameter import parameters_to_ndarrays
from flwr.common.typing import Config
from flwr.server.client_manager import ClientManager

from fl4health.checkpointing.checkpointer import TorchCheckpointer
from fl4health.parameter_exchange.packing_exchanger import FullParameterExchangerWithPacking
from fl4health.parameter_exchange.parameter_packer import ParameterPackerAdaptiveConstraint
from fl4health.reporting.base_reporter import BaseReporter
from fl4health.servers.base_server import FlServer
from fl4health.strategies.fedavg_with_adaptive_constraint import FedAvgWithAdaptiveConstraint


class FedProxServer(FlServer):
    def __init__(
        self,
        client_manager: ClientManager,
        fl_config: Config,
        strategy: FedAvgWithAdaptiveConstraint,
        model: Optional[nn.Module] = None,
        checkpointer: Optional[Union[TorchCheckpointer, Sequence[TorchCheckpointer]]] = None,
        reporters: Sequence[BaseReporter] | None = None,
    ) -> None:
        """
        This is a wrapper class around FlServer for using the FedProx method that enforces that the
        parameter exchanger is a FullParameterExchangerWithPacking of the right type for model rehydration and that
        the strategy is of type FedAvgWithAdaptiveConstraint.

        Args:
            client_manager (ClientManager): Determines the mechanism by which clients are sampled by the server, if
                they are to be sampled at all.
            fl_config (Config): This should be the configuration that was used to setup the federated training.
                In most cases it should be the "source of truth" for how FL training/evaluation should proceed. For
                example, the config used to produce the on_fit_config_fn and on_evaluate_config_fn for the strategy.
                NOTE: This config is DISTINCT from the Flwr server config, which is extremely minimal.
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
            reporters (Sequence[BaseReporter], optional): A sequence of FL4Health reporters which the server should
                send data to before and after each round.
        """
        assert isinstance(
            strategy, FedAvgWithAdaptiveConstraint
        ), "Strategy must be of base type FedAvgWithAdaptiveConstraint"
        parameter_exchanger = FullParameterExchangerWithPacking(ParameterPackerAdaptiveConstraint())
        super().__init__(
            client_manager=client_manager,
            fl_config=fl_config,
            parameter_exchanger=parameter_exchanger,
            model=model,
            strategy=strategy,
            checkpointer=checkpointer,
            reporters=reporters,
        )

    def _hydrate_model_for_checkpointing(self) -> None:
        assert self.server_model is not None, (
            "Model hydration has been called but no server_model is defined to hydrate. The functionality of "
            "_hydrate_model_for_checkpointing can be overridden if checkpointing without a torch architecture is "
            "possible and desired"
        )
        assert self.parameter_exchanger is not None, (
            "Model hydration has been called but no parameter_exchanger is defined to hydrate. The functionality of "
            "_hydrate_model_for_checkpointing can be overridden if checkpointing without a parameter exchanger is "
            "possible and desired"
        )
        # Overriding the standard hydration method to account for the unpacking
        packed_parameters = parameters_to_ndarrays(self.parameters)
        # Don't need the extra loss weight variable for checkpointing.
        assert isinstance(self.parameter_exchanger, FullParameterExchangerWithPacking)
        model_ndarrays, _ = self.parameter_exchanger.unpack_parameters(packed_parameters)
        self.parameter_exchanger.pull_parameters(model_ndarrays, self.server_model)
