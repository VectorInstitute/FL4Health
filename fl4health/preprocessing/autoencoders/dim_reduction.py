from abc import ABC
from pathlib import Path

import torch


class Processing(ABC):
    """
    Abstact class to add processing methods.
    """

    def __init__(
        self,
        checkpointing_path: Path,
    ) -> None:
        self.checkpointing_path = checkpointing_path
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.load_autoencoder()

    def load_autoencoder(self) -> None:
        autoencoder = torch.load(self.checkpointing_path)
        autoencoder.eval()
        self.autoencoder = autoencoder.to(self.device)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"


class AeProcessor(Processing):
    """Transformer processor to encode the data using basic autoencoder."""

    def __init__(
        self,
        checkpointing_path: Path,
    ) -> None:
        super().__init__(checkpointing_path)

    def __call__(self, sample: torch.Tensor) -> torch.Tensor:
        # This transformer is called for the input samples after they are transfered into torch tensors.
        embedding_vector = self.autoencoder.encode(sample.to(self.device))
        return embedding_vector.clone().detach()


class VaeProcessor(Processing):
    """Transformer processor to encode the data using VAE encoder."""

    def __init__(
        self,
        checkpointing_path: Path,
        include_variance: bool = False,
    ) -> None:
        super().__init__(checkpointing_path)
        self.include_variance = include_variance

    def __call__(self, sample: torch.Tensor) -> torch.Tensor:
        # This transformer is called for the input samples after they are transfered into torch tensors.
        mu, logvar = self.autoencoder.encode(sample.to(self.device))
        if self.include_variance:
            return torch.cat(mu.clone().detach(), logvar.clone().detach())
        return mu.clone().detach()


class CvaeFixedConditionProcessor(Processing):
    """Transformer processor to encode the data using CVAE encoder with client-specific condition."""

    def __init__(
        self,
        checkpointing_path: Path,
        condition: torch.Tensor,
        include_variance: bool = False,
    ) -> None:
        self.condition = condition
        self.include_variance = include_variance
        assert (
            self.condition.dim() == 1
        ), f"Error: condition should be a 1D vector instead of a {self.condition.dim()}D tensor."
        super().__init__(checkpointing_path)

    def __call__(self, sample: torch.Tensor) -> torch.Tensor:
        # This transformer is called for the input samples after they are transfered into torch tensors.
        mu, logvar = self.autoencoder.encode(sample.to(self.device), self.condition.to(self.device))
        if self.include_variance:
            return torch.cat(mu.clone().detach(), logvar.clone().detach())
        return mu.clone().detach()
