from pathlib import Path
from typing import Optional, Sequence

import torch
from flwr.common.typing import Config

from fl4health.checkpointing.client_module import ClientCheckpointModule
from fl4health.clients.basic_client import BasicClient
from fl4health.parameter_exchange.parameter_exchanger_base import ParameterExchanger
from fl4health.utils.losses import LossMeterType
from fl4health.utils.metrics import Metric
from fl4health.reporting.metrics import MetricsReporter
from fl4health.parameter_exchange.layer_exchanger import DynamicLayerExchanger
from fl4health.parameter_exchange.parameter_selection_criteria import select_mask_scores
from fl4health.model_bases.masked_model import convert_to_masked_model

class FedPmClient(BasicClient):
    def __init__(
        self,
        data_path: Path,
        metrics: Sequence[Metric],
        device: torch.device,
        loss_meter_type: LossMeterType = LossMeterType.AVERAGE,
        checkpointer: Optional[ClientCheckpointModule] = None,
        metrics_reporter: Optional[MetricsReporter] = None,
    ) -> None:
        super().__init__(
            data_path=data_path,
            metrics=metrics,
            device=device,
            loss_meter_type=loss_meter_type,
            checkpointer=checkpointer,
            metrics_reporter=metrics_reporter,
        )
    
    def setup_client(self, config: Config) -> None:
        super().setup_client(config)
        self.model = convert_to_masked_model(self.model).to(self.device)
    
    def get_parameter_exchanger(self, config: Config) -> ParameterExchanger:
        return DynamicLayerExchanger(layer_selection_function=select_mask_scores)
