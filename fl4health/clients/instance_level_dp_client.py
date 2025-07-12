from collections.abc import Sequence
from pathlib import Path

import torch
from flwr.common.typing import Config
from opacus import PrivacyEngine

from fl4health.checkpointing.client_module import ClientCheckpointAndStateModule
from fl4health.clients.basic_client import BasicClient
from fl4health.metrics.base_metrics import Metric
from fl4health.reporting.base_reporter import BaseReporter
from fl4health.utils.config import narrow_dict_type
from fl4health.utils.losses import LossMeterType
from fl4health.utils.privacy_utilities import privacy_validate_and_fix_modules


class InstanceLevelDpClient(BasicClient):
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
        Client for Instance/Record level Differentially Private Federated Averaging.

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
        self.clipping_bound: float
        self.noise_multiplier: float

    def setup_client(self, config: Config) -> None:
        """
        Performs the same flow as ``BasicClient`` to setup a client. This functionality straps on a processing of two
        configuration variables ``self.clipping_bound`` and ``self.noise_multiplier``. The last step is to do some
        processing of the model and optimizers with Opacus to make them DP compatible and to setup the privacy engine
        used for privacy accounting. This is done with the ``setup_opacus_objects`` function.

        Args:
            config (Config): Configurations sent by the server to allow for customization of this functions behavior.
        """
        # Ensure that clipping bound and noise multiplier is present in config
        # Set attributes to be used when setting DP training
        self.clipping_bound = narrow_dict_type(config, "clipping_bound", float)
        self.noise_multiplier = narrow_dict_type(config, "noise_multiplier", float)

        # Do basic client setup
        super().setup_client(config)

        # Configure DP training
        self.setup_opacus_objects(config)

    def setup_opacus_objects(self, config: Config) -> None:
        """
        Validates and potentially fixes the PyTorch model of the client to be compatible with Opacus and privacy
        mechanisms, sets up the privacy engine of Opacus using the model, optimizer, dataloaders etc. for DP training.

        Args:
            config (Config): Configurations sent by the server to allow for customization of this functions behavior.
        """
        # Validate that the model layers are compatible with privacy mechanisms in Opacus and try to replace the layers
        # with compatible ones if necessary.
        self.model, reinitialize_optimizer = privacy_validate_and_fix_modules(self.model)

        # If we have fixed the model by changing out layers (and therefore parameters), we need to update the optimizer
        # parameters to coincide with this fixed model. **NOTE**: It is not done in make_private!
        if reinitialize_optimizer:
            self.set_optimizer(config)

        # Create DP training objects
        privacy_engine = PrivacyEngine()
        # NOTE: that Opacus make private is NOT idempotent
        self.model, optimizer, self.train_loader = privacy_engine.make_private(
            module=self.model,
            optimizer=self.optimizers["global"],
            data_loader=self.train_loader,
            noise_multiplier=self.noise_multiplier,
            max_grad_norm=self.clipping_bound,
            clipping="flat",
        )

        self.optimizers = {"global": optimizer}
