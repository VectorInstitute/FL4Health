from pathlib import Path
from typing import Optional, Sequence

import torch
from flwr.common.typing import Config

from fl4health.checkpointing.client_module import ClientCheckpointModule
from fl4health.clients.basic_client import BasicClient
from fl4health.model_bases.masked_layers import convert_to_masked_model
from fl4health.parameter_exchange.fedpm_exchanger import FedPmExchanger
from fl4health.parameter_exchange.parameter_exchanger_base import ParameterExchanger
from fl4health.reporting.base_reporter import BaseReporter
from fl4health.utils.config import narrow_dict_type
from fl4health.utils.losses import LossMeterType
from fl4health.utils.metrics import Metric


class FedPmClient(BasicClient):
    def __init__(
        self,
        data_path: Path,
        metrics: Sequence[Metric],
        device: torch.device,
        loss_meter_type: LossMeterType = LossMeterType.AVERAGE,
        checkpointer: Optional[ClientCheckpointModule] = None,
        reporters: Sequence[BaseReporter] | None = None,
    ) -> None:
        super().__init__(
            data_path=data_path,
            metrics=metrics,
            device=device,
            loss_meter_type=loss_meter_type,
            checkpointer=checkpointer,
            reporters=reporters,
        )

    def setup_client(self, config: Config) -> None:
        super().setup_client(config)
        # Convert self.model to a masked model unless it is specified in the config
        # file that the model is already a masked model.
        is_masked_model = narrow_dict_type(config, "is_masked_model", bool)
        if not is_masked_model:
            self.model = convert_to_masked_model(self.model).to(self.device)

    def get_parameter_exchanger(self, config: Config) -> ParameterExchanger:
        return FedPmExchanger()
