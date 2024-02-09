from abc import ABC
from pathlib import Path

import torch


class AutoEncoderProcessing(ABC):
    def __init__(
        self,
        checkpointing_path: Path,
    ) -> None:
        """
        Abstract class for processors that work with a pre-trained and saved autoencoder model.

        Args:
            checkpointing_path (Path): Path to the saved model.
        """
        self.checkpointing_path = checkpointing_path
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.load_autoencoder()

    def load_autoencoder(self) -> None:
        autoencoder = torch.load(self.checkpointing_path)
        autoencoder.eval()
        self.autoencoder = autoencoder.to(self.device)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"


class AeProcessor(AutoEncoderProcessing):
    """
    Transformer processor to encode the data using basic autoencoder.
    """

    def __call__(self, sample: torch.Tensor) -> torch.Tensor:
        # This transformer is called for the input samples after they are transfered into torch tensors.
        embedding_vector = self.autoencoder.encode(sample.to(self.device))
        return embedding_vector.clone().detach()


class VaeProcessor(AutoEncoderProcessing):
    def __init__(
        self,
        checkpointing_path: Path,
        return_mu: bool = False,
    ) -> None:
        """
        Transformer processor to encode the data using VAE encoder.

        Args:
            checkpointing_path (Path): Path to the saved model.
            return_mu (bool, optional): If true, only mu is returned. Defaults to False.
        """
        super().__init__(checkpointing_path)
        self.return_mu = return_mu

    def __call__(self, sample: torch.Tensor) -> torch.Tensor:
        # This transformer is called for the input samples after they are transfered into torch tensors.
        mu, logvar = self.autoencoder.encode(sample.to(self.device))
        if self.return_mu:
            return mu.clone().detach()
        # By default returns cat(mu,logvar).
        # Contatation is performed on the last dimention which for both the batched data and single data
        # is the latent space dimention.
        return torch.cat((mu.clone().detach(), logvar.clone().detach()), dim=-1)


class CvaeFixedConditionProcessor(AutoEncoderProcessing):
    """Transformer processor to encode the data using CVAE encoder with client-specific condition."""

    def __init__(
        self,
        checkpointing_path: Path,
        condition: torch.Tensor,
        return_mu: bool = False,
        batch_size: int = 1,
    ) -> None:
        super().__init__(checkpointing_path)
        self.condition = condition
        self.return_mu = return_mu
        assert (
            self.condition.dim() == 1
        ), f"Error: condition should be a 1D vector instead of a {self.condition.dim()}D tensor."
        # Assuming data is "batch first"
        # If we are processing a batch of data, condition should be repeated for each sample.
        if batch_size > 1:
            self.condition = self.condition.repeat(batch_size, 1)

    def __call__(self, sample: torch.Tensor) -> torch.Tensor:
        # This transformer is called for the input samples after they are transfered into torch tensors.
        mu, logvar = self.autoencoder.encode(sample.to(self.device), self.condition.to(self.device))
        if self.return_mu:
            return mu.clone().detach()
        # By default returns cat(mu,logvar)
        # Contatation is performed on the last dimention which for both the batched data and single data
        # is the latent space dimention.
        return torch.cat((mu.clone().detach(), logvar.clone().detach()), dim=-1)


class CvaeNonFixedConditionProcessor(AutoEncoderProcessing):
    def __init__(
        self,
        checkpointing_path: Path,
        return_mu: bool = False,
    ) -> None:
        """Transformer processor to encode the data using CVAE encoder with non-fixed condition,
        that is each data sample can have a specific condition.

        Args:
            checkpointing_path (Path): Path to the saved model.
            return_mu (bool, optional): If true, only mu is returned. Defaults to False.
        """
        super().__init__(checkpointing_path)
        self.return_mu = return_mu

    def __call__(self, sample: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """Performs encoding.

        Args:
            sample (torch.Tensor): A single data sample or a batch of data.
            condition (torch.Tensor): A single condition for the given sample,
                or a batch of condition for a batch of data.

        Returns:
            torch.Tensor: Encoded sample(s).
        """
        # This transformer is called for the input samples after they are transfered into torch tensors.
        # We assume condition and data are "batch first".
        if condition.size(0) > 1:  # If condition is a batch
            assert condition.size(0) == sample.size(
                0
            ), f"Error: Condition shape: {condition.shape} does not match the data shape: {sample.shape}"
        mu, logvar = self.autoencoder.encode(sample.to(self.device), condition.to(self.device))
        if self.return_mu:
            return mu.clone().detach()
        # By default returns cat(mu,logvar)
        # Contatation is performed on the last dimention which for both the batched data and single data
        # is the latent space dimention.
        return torch.cat((mu.clone().detach(), logvar.clone().detach()), dim=-1)
