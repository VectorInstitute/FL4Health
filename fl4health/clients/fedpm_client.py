from collections.abc import Sequence
from pathlib import Path

import torch
from flwr.common.typing import Config

from fl4health.checkpointing.client_module import ClientCheckpointAndStateModule
from fl4health.clients.basic_client import BasicClient
from fl4health.metrics.base_metrics import Metric
from fl4health.model_bases.masked_layers.masked_layers_utils import convert_to_masked_model
from fl4health.parameter_exchange.fedpm_exchanger import FedPmExchanger
from fl4health.parameter_exchange.parameter_exchanger_base import ParameterExchanger
from fl4health.reporting.base_reporter import BaseReporter
from fl4health.utils.config import narrow_dict_type
from fl4health.utils.losses import LossMeterType


class FedPmClient(BasicClient):
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
        Client implementing the FedPM algorithm (https://arxiv.org/pdf/2209.15328).

        FedPM is a recent sparse, communication efficient approach to federated learning. The method has been shown to
        have exceptional information compression while maintaining good performance. Interestingly, it is also
        connected to the Lottery Ticket Hypothesis. Training on the client-side is effectively the same as
        ``BasicClient``. The two components that change are ensuring that the model to be training is a Masked Model
        compatible with FedPM (or to convert it to one). Second, we use the FedPM exchanger to facilitate exchange
        with the server.

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

    def setup_client(self, config: Config) -> None:
        """
        Performs the same setup as ``BasicClient``, but adds on the possibility of converting an ordinary model to a
        masked model compatible with ``FedPM``.

        Args:
            config (Config): Configuration specifying all of the required parameters for training.
        """
        super().setup_client(config)
        # Convert self.model to a masked model unless it is specified in the config
        # file that the model is already a masked model.
        is_masked_model = narrow_dict_type(config, "is_masked_model", bool)
        if not is_masked_model:
            self.model = convert_to_masked_model(self.model).to(self.device)

    def get_parameter_exchanger(self, config: Config) -> ParameterExchanger:
        """
        Forces the client to use the ``FedPmExchanger``.

        Args:
            config (Config): Configuration specifying all of the required parameters for training.

        Returns:
            (ParameterExchanger): returns a ``FedPmExchanger`` to facilitate exchange properly
        """
        return FedPmExchanger()
