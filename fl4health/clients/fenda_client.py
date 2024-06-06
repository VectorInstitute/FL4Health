from pathlib import Path
from typing import Optional, Sequence

import torch
from flwr.common.typing import Config

from fl4health.checkpointing.client_module import ClientCheckpointModule
from fl4health.clients.basic_client import BasicClient
from fl4health.model_bases.fenda_base import FendaModel
from fl4health.parameter_exchange.layer_exchanger import FixedLayerExchanger
from fl4health.parameter_exchange.parameter_exchanger_base import ParameterExchanger
from fl4health.utils.losses import LossMeterType
from fl4health.utils.metrics import Metric


class FendaClient(BasicClient):
    def __init__(
        self,
        data_path: Path,
        metrics: Sequence[Metric],
        device: torch.device,
        loss_meter_type: LossMeterType = LossMeterType.AVERAGE,
        checkpointer: Optional[ClientCheckpointModule] = None,
    ) -> None:
        super().__init__(
            data_path=data_path,
            metrics=metrics,
            device=device,
            loss_meter_type=loss_meter_type,
            checkpointer=checkpointer,
        )
        """
        This client is used to perform client-side training associated with the FENDA method described in
        https://arxiv.org/pdf/2309.16825. The approach splits a model being trained into parallel feature extractors
        whose latent feature spaces are then further processed by a classification head. The global feature extractor
        is federally trained with FedAvg and the local feature extractor and classification head are exclusively
        trained locally. This is closely related (and essentially an ablation of the PerFCL method).
        Args:
            data_path (Path): Path to the data directory.
            metrics (Sequence[Metric]): List of metrics to be used for evaluation.
            device (torch.device): Device to be used for training.
            loss_meter_type (LossMeterType, optional): Type of loss meter to be used. Defaults to
                LossMeterType.AVERAGE.
            checkpointer (Optional[ClientCheckpointModule], optional): Checkpointer module defining when and how to
                do checkpointing during client-side training. No checkpointing is done if not provided. Defaults to
                None.
        """

    def get_parameter_exchanger(self, config: Config) -> ParameterExchanger:
        assert isinstance(self.model, FendaModel)
        return FixedLayerExchanger(self.model.layers_to_exchange())
