from logging import WARNING
from pathlib import Path
from typing import Optional, Sequence

import torch
from flwr.common.logger import log
from flwr.common.typing import Config
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator

from fl4health.checkpointing.checkpointer import TorchCheckpointer
from fl4health.clients.basic_client import BasicClient
from fl4health.utils.losses import LossMeterType
from fl4health.utils.metrics import Metric


class InstanceLevelPrivacyClient(BasicClient):
    """
    Client for Instance Differentially Private Federated Averaging
    """

    def __init__(
        self,
        data_path: Path,
        metrics: Sequence[Metric],
        device: torch.device,
        loss_meter_type: LossMeterType = LossMeterType.AVERAGE,
        checkpointer: Optional[TorchCheckpointer] = None,
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
        self.setup_opacus_objects()

    def setup_opacus_objects(self) -> None:
        # Validate that the model layers are compatible with privacy mechanisms in Opacus and try to replace the layers
        # with compatible ones if necessary.
        errors = ModuleValidator.validate(self.model, strict=False)
        if len(errors) != 0:
            for error in errors:
                log(WARNING, f"Opacus error: {error}")
            self.model = ModuleValidator.fix(self.model)

        # Create DP training objects
        privacy_engine = PrivacyEngine()
        # NOTE: that Opacus make private is NOT idempotent
        self.model, self.optimizer, self.train_loader = privacy_engine.make_private(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=self.train_loader,
            noise_multiplier=self.noise_multiplier,
            max_grad_norm=self.clipping_bound,
            clipping="flat",
        )
