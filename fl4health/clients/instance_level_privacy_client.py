from logging import WARNING
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from flwr.common.logger import log
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from torch.utils.data import DataLoader

from fl4health.clients.numpy_fl_client import NumpyFlClient


class InstanceLevelPrivacyClient(NumpyFlClient):
    """
    Client for Instance Differentially Private Federated Averaging
    """

    def __init__(self, data_path: Path, device: torch.device) -> None:
        super().__init__(data_path, device)

        self.train_loader: DataLoader
        self.model: nn.Module
        self.optimizer: torch.optim.Optimizer
        self.noise_multiplier: float
        self.clipping_bound: float
        self.num_examples: Dict[str, int]

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
