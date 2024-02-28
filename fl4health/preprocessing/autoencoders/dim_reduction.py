from abc import ABC
from pathlib import Path

import torch


class AutoEncoderProcessing(ABC):
    def __init__(
        self,
        checkpointing_path: Path,
        device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    ) -> None:
        """Abstract class for processors that work with a pre-trained and saved autoencoder model.

        Args:
            checkpointing_path (Path): Path to the saved model.
            device (torch.device, optional): Device indicator for where to send the model and data samples
            for preprocessing.
        """

        self.checkpointing_path = checkpointing_path
        self.device = device
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
        # This transformer is called for the input samples after they are transferred into torch tensors.
        embedding_vector = self.autoencoder.encode(sample.to(self.device))
        return embedding_vector.clone().detach()


class VaeProcessor(AutoEncoderProcessing):
    def __init__(
        self,
        checkpointing_path: Path,
        device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        return_mu_only: bool = False,
    ) -> None:
        """
        Transformer processor to encode the data using VAE encoder.

        Args:
            checkpointing_path (Path): Path to the saved model.
            device (torch.device, optional): Device indicator for where to send the model and data samples
            for preprocessing.
            return_mu_only (bool, optional): If true, only mu is returned. Defaults to False.
        """
        super().__init__(checkpointing_path, device)
        self.return_mu_only = return_mu_only

    def __call__(self, sample: torch.Tensor) -> torch.Tensor:
        # This transformer is called for the input samples after they are transferred into torch tensors.
        mu, logvar = self.autoencoder.encode(sample.to(self.device))
        if self.return_mu_only:
            return mu.clone().detach()
        # By default returns cat(mu,logvar).
        # Concatenation is performed on the last dimension which for both the batched data and single data
        # is the latent space dimension.
        return torch.cat((mu.clone().detach(), logvar.clone().detach()), dim=-1)


class CvaeFixedConditionProcessor(AutoEncoderProcessing):
    """Transformer processor to encode the data using CVAE encoder with client-specific condition."""

    def __init__(
        self,
        checkpointing_path: Path,
        condition: torch.Tensor,
        device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        return_mu_only: bool = False,
    ) -> None:
        """Transformer processor to encode the data using a CVAE encoder with a fixed condition.

        Args:
            checkpointing_path (Path): Path to the saved model.
            condition (torch.Tensor): Fixed condition tensor.
            device (torch.device, optional):  Device indicator for where to send the model and data samples
            for preprocessing.
            return_mu_only (bool, optional): If true, only mu is returned. Defaults to False.
        """
        super().__init__(checkpointing_path, device)
        self.condition = condition
        self.return_mu_only = return_mu_only
        assert (
            self.condition.dim() == 1
        ), f"Error: condition should be a 1D vector instead of a {self.condition.dim()}D tensor."

    def __call__(self, sample: torch.Tensor) -> torch.Tensor:
        # Assuming batch is the first dimension
        if sample.dim() == 1:
            batch_condition = self.condition
        else:
            # If we are processing a batch of data, condition should be repeated for each sample.
            sample_batch_size = sample.shape[0]
            batch_condition = self.condition.repeat(sample_batch_size, 1)
        # This transformer is called for the input samples after they are transformed into torch tensors.
        mu, logvar = self.autoencoder.encode(sample.to(self.device), batch_condition.to(self.device))
        if self.return_mu_only:
            return mu.clone().detach()
        # By default returns cat(mu,logvar)
        # Concatenation is performed on the last dimension which for both the batched data and single data
        # is the latent space dimension.
        return torch.cat((mu.clone().detach(), logvar.clone().detach()), dim=-1)


class CvaeVariableConditionProcessor(AutoEncoderProcessing):
    def __init__(
        self,
        checkpointing_path: Path,
        device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        return_mu_only: bool = False,
    ) -> None:
        """
        Transformer processor to encode the data using CVAE encoder with variable condition,
        that is each data sample can have a specific condition.

        Args:
            checkpointing_path (Path): Path to the saved model.
            device (torch.device, optional): Device indicator for where to send the model and data samples
            for preprocessing.
            return_mu_only (bool, optional): If true, only mu is returned. Defaults to False.
        """
        super().__init__(checkpointing_path, device)
        self.return_mu_only = return_mu_only

    def __call__(self, sample: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """Performs encoding.

        Args:
            sample (torch.Tensor): A single data sample or a batch of data.
            condition (torch.Tensor): A single condition for the given sample,
                or a batch of condition for a batch of data.

        Returns:
            torch.Tensor: Encoded sample(s).
        """
        # This transformer is called for the input samples after they are transformed into torch tensors.
        # We assume condition and data are "batch first".
        if condition.size(0) > 1:  # If condition is a batch
            assert condition.size(0) == sample.size(
                0
            ), f"Error: Condition shape: {condition.shape} does not match the data shape: {sample.shape}"
        mu, logvar = self.autoencoder.encode(sample.to(self.device), condition.to(self.device))
        if self.return_mu_only:
            return mu.clone().detach()
        # By default returns cat(mu,logvar)
        # Concatenation is performed on the last dimension which for both the batched data and single data
        # is the latent space dimension.
        return torch.cat((mu.clone().detach(), logvar.clone().detach()), dim=-1)
