from abc import ABC, abstractmethod
from typing import Tuple, Optional
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np


class Processing(ABC):
    """
    Abstact class to add processing methods.
    """

    def __init__(
        self,
        checkpoint_path: Path,
    ) -> None:
        self.autoencoder = self.load_autoencoder(checkpoint_path)

    def load_autoencoder(self, checkpoint_path: Path) -> nn.Module:
        autoencoder = torch.load(checkpoint_path)
        autoencoder.eval()
        return autoencoder

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"


class VAEProcessor(Processing):
    """Transformer processor to encode the data using VAE encoder."""

    def __init__(
        self,
        checkpoint_path: Path,
    ) -> None:
        super().__init__(checkpoint_path)

    def __call__(self, sample: np.ndarray) -> torch.Tensor:
        mu, logvar = self.autoencoder.encode(sample)
        return mu


class LabelConditionedProcessor(Processing):
    """Transformer processor to encode the data using CVAE encoder, with the labels as conditions."""

    def __init__(
        self,
        checkpoint_path: Path,
    ) -> None:
        super().__init__(checkpoint_path)

    def __call__(self, sample: np.ndarray, target: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        condition_vector = self.autoencoder.one_hot(torch.tensor(target))
        mu, logvar = self.autoencoder.encode(sample, condition_vector)
        return mu, torch.from_numpy(target)


class ClientConditionedProcessor(Processing):
    """Transformer processor to encode the data using CVAE encoder with client-specific condition."""

    def __init__(
        self,
        checkpoint_path: Path,
        condition: int,
    ) -> None:
        self.condition = condition
        super().__init__(checkpoint_path)

    def __call__(self, sample: np.ndarray) -> torch.Tensor:
        assert self.condition != None
        condition_vector = self.autoencoder.one_hot(torch.tensor(self.condition))
        mu, logvar = self.autoencoder.encode(sample, condition_vector)
        return mu
