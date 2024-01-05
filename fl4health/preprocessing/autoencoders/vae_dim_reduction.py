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
        self.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.load_autoencoder()

    def load_autoencoder(self) -> None:
        autoencoder = torch.load(self.checkpointing_path)
        autoencoder.eval()
        self.autoencoder = autoencoder.to(self.DEVICE)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"


class AEProcessor(Processing):
    """Transformer processor to encode the data using basic autoencoder."""

    def __init__(
        self,
        checkpointing_path: Path,
    ) -> None:
        super().__init__(checkpointing_path)

    def __call__(self, sample: torch.Tensor) -> torch.Tensor:
        # This transformer is called for the input samples after they are transfered into torch tensors.
        embedding_vector = self.autoencoder.encode(sample.to(self.DEVICE))
        return embedding_vector.clone().detach()


class VAEProcessor(Processing):
    """Transformer processor to encode the data using VAE encoder."""

    def __init__(
        self,
        checkpointing_path: Path,
    ) -> None:
        super().__init__(checkpointing_path)

    def __call__(self, sample: torch.Tensor) -> torch.Tensor:
        # This transformer is called for the input samples after they are transfered into torch tensors.
        mu, logvar = self.autoencoder.encode(sample.to(self.DEVICE))
        return mu.clone().detach()


class ClientConditionedProcessor(Processing):
    """Transformer processor to encode the data using CVAE encoder with client-specific condition."""

    def __init__(
        self,
        checkpointing_path: Path,
        condition: int,
    ) -> None:
        self.condition = condition
        super().__init__(checkpointing_path)

    def __call__(self, sample: torch.Tensor) -> torch.Tensor:
        # This transformer is called for the input samples after they are transfered into toch tensors.
        condition_vector = self.autoencoder.one_hot(
            torch.tensor(self.condition).to(self.DEVICE),
        )
        mu, logvar = self.autoencoder.encode(sample.to(self.DEVICE), condition_vector)
        return mu.clone().detach()
