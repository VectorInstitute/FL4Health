from collections.abc import Sequence
from pathlib import Path

import torch
from flwr.common.typing import Config

from fl4health.checkpointing.client_module import ClientCheckpointAndStateModule
from fl4health.clients.basic_client import BasicClient
from fl4health.metrics.base_metrics import Metric
from fl4health.model_bases.fenda_base import FendaModel
from fl4health.parameter_exchange.layer_exchanger import FixedLayerExchanger
from fl4health.parameter_exchange.parameter_exchanger_base import ParameterExchanger
from fl4health.reporting.base_reporter import BaseReporter
from fl4health.utils.losses import LossMeterType


class FendaClient(BasicClient):
    def __init__(
        self,
        data_path: Path,
        metrics: Sequence[Metric],
        device: torch.device,
        loss_meter_type: LossMeterType = LossMeterType.AVERAGE,
        checkpoint_and_state_module: ClientCheckpointAndStateModule | None = None,
        reporters: Sequence[BaseReporter] | None = None,
        progress_bar: bool = False,
        client_name: str | None = None,
    ) -> None:
        """
        This client is used to perform client-side training associated with the FENDA method described in
        https://arxiv.org/pdf/2309.16825.

        The approach splits a model being trained into parallel feature extractors whose latent feature spaces are
        then further processed by a classification head. The global feature extractor is federally trained with FedAvg
        and the local feature extractor and classification head are exclusively trained locally. This is closely
        related (and is essentially an ablation of) the PerFCL method.

        Args:
            data_path (Path): path to the data to be used to load the data for client-side training.
            metrics (Sequence[Metric]): Metrics to be computed based on the labels and predictions of the client model.
            device (torch.device): Device indicator for where to send the model, batches, labels etc. Often "cpu" or
                "cuda".
            loss_meter_type (LossMeterType, optional): Type of meter used to track and compute the losses over
                each batch. Defaults to ``LossMeterType.AVERAGE``.
            checkpoint_and_state_module (ClientCheckpointAndStateModule | None, optional): A module meant to handle
                both checkpointing and state saving. The module, and its underlying model and state checkpointing
                components will determine when and how to do checkpointing during client-side training.
                No checkpointing (state or model) is done if not provided. Defaults to None.
            reporters (Sequence[BaseReporter] | None, optional): A sequence of FL4Health reporters which the client
                should send data to. Defaults to None.
            progress_bar (bool, optional): Whether or not to display a progress bar during client training and
                validation. Uses ``tqdm``. Defaults to False
            client_name (str | None, optional): An optional client name that uniquely identifies a client.
                If not passed, a hash is randomly generated. Client state will use this as part of its state file
                name. Defaults to None.
        """
        super().__init__(
            data_path=data_path,
            metrics=metrics,
            device=device,
            loss_meter_type=loss_meter_type,
            checkpoint_and_state_module=checkpoint_and_state_module,
            reporters=reporters,
            progress_bar=progress_bar,
            client_name=client_name,
        )

    def get_parameter_exchanger(self, config: Config) -> ParameterExchanger:
        assert isinstance(self.model, FendaModel)
        return FixedLayerExchanger(self.model.layers_to_exchange())
