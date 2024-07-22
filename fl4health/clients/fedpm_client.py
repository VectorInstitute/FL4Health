from pathlib import Path
from typing import Optional, Sequence

import torch
from flwr.common.typing import Config

from fl4health.checkpointing.client_module import ClientCheckpointModule
from fl4health.clients.basic_client import BasicClient
from fl4health.model_bases.masked_layers import convert_to_masked_model
from fl4health.parameter_exchange.fedpm_exchanger import FedPmExchanger
from fl4health.parameter_exchange.parameter_exchanger_base import ParameterExchanger
from fl4health.parameter_exchange.parameter_selection_criteria import select_scores_and_sample_masks
from fl4health.reporting.metrics import MetricsReporter
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
        # Convert self.model to a masked model unless it is specified in the config
        # file that the model is already a masked model.
        if "is_masked_model" in config.keys():
            is_masked_model = self.narrow_config_type(config, "is_masked_model", bool)
            if not is_masked_model:
                self.model = convert_to_masked_model(self.model).to(self.device)
        else:
            self.model = convert_to_masked_model(self.model).to(self.device)

    def get_parameter_exchanger(self, config: Config) -> ParameterExchanger:
        return FedPmExchanger(layer_selection_function=select_scores_and_sample_masks)
