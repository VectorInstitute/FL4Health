from fl4health.clients.basic_client import BasicClient
from pathlib import Path
from typing import Sequence, Optional, Tuple, Dict
from fl4health.utils.metrics import Metric, MetricMeter, MetricMeterManager, MetricMeterType
import torch
import torch.nn as nn
from fl4health.utils.losses import Losses, LossMeter, LossMeterType
from fl4health.checkpointing.checkpointer import TorchCheckpointer
from flwr.common.typing import Config, NDArrays, Scalar
from logging import DEBUG, INFO, WARNING
from flwr.common.logger import log



class SecureAggregationClient(BasicClient):
    
    def __init__(
        self,
        data_path: Path,
        metrics: Sequence[Metric],
        device: torch.device,
        loss_meter_type: LossMeterType = LossMeterType.AVERAGE,
        metric_meter_type: MetricMeterType = MetricMeterType.AVERAGE,
        checkpointer: Optional[TorchCheckpointer] = None,
    ) -> None:
        
        super().__init__(data_path, metrics, device, loss_meter_type, metric_meter_type, checkpointer)

    # def fit(self, config: Config, num_epochs: int) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
    #     pass

    def get_properties(self, config: Config) -> Dict[str, Scalar]:
        if not self.initialized:
            self.setup_client(config)
        return {"num_train_samples": self.num_train_samples, "num_val_samples": self.num_val_samples}
