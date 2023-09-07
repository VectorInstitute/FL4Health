
from pathlib import Path
from typing import Optional, Sequence

import torch
import torch.nn as nn
from flwr.common.typing import Config 

from fl4health.checkpointing.checkpointer import TorchCheckpointer
from fl4health.clients.basic_client import BasicClient 
from fl4health.parameter_exchange.layer_exchanger import FixedLayerExchanger
from fl4health.parameter_exchange.parameter_exchanger_base import ParameterExchanger
from fl4health.utils.metrics import Metric
from fl4health.model_bases.fenda_base import FendaModel

class FendaClient(BasicClient): 
    def __init__(
            self,
            data_path: Path,
            metrics: Sequence[Metric],
            device: torch.device,
            meter_type: str = "average",
            use_wandb_reporter: bool = False,
            checkpointer: Optional[TorchCheckpointer] = None
    ) -> None: 
        super().__init__(
            data_path=data_path,
            metrics=metrics,
            device=device,
            meter_type=meter_type,
            use_wandb_reporter=use_wandb_reporter,
            checkpointer=checkpointer
        )
    
    def get_parameter_exchanger(self, config: Config, model: nn.Module) -> ParameterExchanger:
        assert isinstance(model, FendaModel)
        return FixedLayerExchanger(model.layers_to_exchange()) 
