from pathlib import Path
from typing import Optional, Sequence

import torch
from flwr.common.typing import Config
from opacus import PrivacyEngine

from fl4health.checkpointing.client_module import ClientCheckpointModule
from fl4health.clients.basic_client import BasicClient
from fl4health.utils.losses import LossMeterType
from fl4health.utils.metrics import Metric
from fl4health.utils.privacy_utilities import privacy_validate_and_fix_modules


class InstanceLevelDpClient(BasicClient):
    """
    Client for Instance/Record level Differentially Private Federated Averaging
    """

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
        self.clipping_bound: float
        self.noise_multiplier: float

    def setup_client(self, config: Config) -> None:
        # Ensure that clipping bound and noise multiplier is present in config
        # Set attributes to be used when setting DP training
        self.clipping_bound = self.narrow_config_type(config, "clipping_bound", float)
        self.noise_multiplier = self.narrow_config_type(config, "noise_multiplier", float)

        # Do basic client setup
        super().setup_client(config)

        # Configure DP training
        self.setup_opacus_objects(config)

    def setup_opacus_objects(self, config: Config) -> None:
        # Validate that the model layers are compatible with privacy mechanisms in Opacus and try to replace the layers
        # with compatible ones if necessary.
        self.model, reinitialize_optimizer = privacy_validate_and_fix_modules(self.model)

        # If we have fixed the model by changing out layers (and therefore parameters), we need to update the optimizer
        # parameters to coincide with this fixed model. NOTE: It is not done in make_private!
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
