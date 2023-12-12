from abc import ABC
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

from fl4health.model_bases.autoencoders_base import ConditionalVAE, VarioationalAE


class Processing(ABC):
    """
    Abstact class to add processing methods.
    """

    def __init__(
        self,
        checkpointing_path: Path,
    ) -> None:
        self.checkpointing_path = checkpointing_path
        self.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def load_VAE(self) -> VarioationalAE:
        autoencoder = torch.load(self.checkpointing_path)
        autoencoder.eval()
        return autoencoder

    def load_CVAE(self) -> ConditionalVAE:
        autoencoder = torch.load(self.checkpointing_path)
        autoencoder.eval()
        return autoencoder

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"


class VAEProcessor(Processing):
    """Transformer processor to encode the data using VAE encoder."""

    def __init__(
        self,
        checkpointing_path: Path,
    ) -> None:
        super().__init__(checkpointing_path)
        self.autoencoder: VarioationalAE = self.load_VAE()

    def __call__(self, sample: torch.Tensor) -> torch.Tensor:
        # This transformer is called for the input samples after they are transfered into toch tensors.
        mu, logvar = self.autoencoder.encode(sample)
        return mu.clone().detach()


class LabelConditionedProcessor(Processing):
    """Transformer processor to encode the data using CVAE encoder, with the labels as conditions."""

    def __init__(
        self,
        checkpointing_path: Path,
    ) -> None:
        super().__init__(checkpointing_path)
        self.autoencoder: ConditionalVAE = self.load_CVAE()

    def __call__(self, sample: np.ndarray, target: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        condition_vector = self.autoencoder.one_hot(torch.tensor(target).to(self.DEVICE))
        mu, logvar = self.autoencoder.encode(torch.tensor(sample).to(self.DEVICE), condition_vector)
        return mu.clone().detach(), torch.from_numpy(target)


class ClientConditionedProcessor(Processing):
    """Transformer processor to encode the data using CVAE encoder with client-specific condition."""

    def __init__(
        self,
        checkpointing_path: Path,
        condition: int,
    ) -> None:
        self.condition = condition
        super().__init__(checkpointing_path)
        self.autoencoder: ConditionalVAE = self.load_CVAE()

    def __call__(self, sample: torch.Tensor) -> torch.Tensor:
        # This transformer is called for the input samples after they are transfered into toch tensors.
        condition_vector = self.autoencoder.one_hot(
            torch.tensor(self.condition).to(self.DEVICE),
        )
        mu, logvar = self.autoencoder.encode(sample.to(self.DEVICE), condition_vector)
        return mu.clone().detach()
