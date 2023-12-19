from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.nn.modules.loss import _Loss

from fl4health.pipeline.loss import VAE_loss


class AEPipeline:
    def load_autoencoder(self, model_path: Path) -> None:
        self.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        autoencoder = torch.load(model_path)
        autoencoder.eval()
        self.autoencoder = autoencoder.to(self.DEVICE)

    def training_transform(self, sample: np.ndarray, target: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """Training transformer for basic autoencoders and variational autoencoders
        which replaces the traget with the sample.

        Args:
            sample (np.ndarray): data sample
            target (np.ndarray): data target

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Proper format for training a basic autoencoders
            or variational autoencoders.
        """
        return torch.from_numpy(sample), torch.from_numpy(sample)

    def dim_reduction_transform(self, sample: torch.Tensor) -> torch.Tensor:
        """Data transformer that encodes the data samples using the encoder.

        Args:
            sample (torch.Tensor): data sample.

        Returns:
            torch.Tensor: encoded data sample.
        """
        # This transformer is called for the input samples after they are transfered into torch tensors.
        embedding_vector = self.autoencoder.encode(sample.to(self.DEVICE))
        return embedding_vector.clone().detach()

    def get_dim_reduce_transform(self, data_transform: transforms, model_path: Path) -> transforms:
        """Loads and autoencoder based model and uses it to create a dimensionality reduction data transformer,
        which is added after the user-defined data transformers.

        Args:
            data_transform (transforms): user-defined data transformers
            model_path (Path): Path to the saved autoencoder based model.

        Returns:
            transforms: dimesionality reduction transformer added to the user-defined data transformers.
        """
        self.load_autoencoder(model_path)
        return transforms.Compose([data_transform, self.dim_reduction_transform])

    def get_AE_loss(self, base_loss: _Loss, latent_dim: int) -> _Loss:
        return base_loss


class VAEPipeline(AEPipeline):
    def dim_reduction_transform(self, sample: torch.Tensor) -> torch.Tensor:
        # This transformer is called for the input samples after they are transfered into torch tensors.
        mu, logvar = self.autoencoder.encode(sample.to(self.DEVICE))
        return mu.clone().detach()

    def get_AE_loss(self, base_loss: _Loss, latent_dim: int) -> _Loss:
        return VAE_loss(latent_dim, base_loss)


class CVAEPipeline(AEPipeline):
    def __init__(self, condition: str = "label") -> None:
        self.condition = condition

    def dim_reduction_transform(self, sample: torch.Tensor) -> torch.Tensor:
        # This transformer is called for the input samples after they are transfered into toch tensors.
        condition_vector = self.autoencoder.one_hot(
            torch.tensor(int(self.condition)).to(self.DEVICE),
        )
        mu, logvar = self.autoencoder.encode(sample.to(self.DEVICE), condition_vector)
        return mu.clone().detach()

    def training_transform(self, sample: np.ndarray, target: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """This is a data_target transformer that trainsforms the input data into a suitable form
        for training a CVAE. The input format of could be either ([sample, int], sample)
        or ([sample, target], sample) based on the condition type.

        Args:
            sample (np.ndarray): data sample
            target (np.ndarray): data target

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Input to the encoder.
        """
        if self.condition.isdigit():
            # Custom condition from the client
            return torch.from_numpy(
                np.concatenate((sample, torch.tensor(int(self.condition)).numpy()), axis=None)
            ), torch.from_numpy(sample)
        else:
            return torch.from_numpy(np.concatenate((sample, target), axis=None)), torch.from_numpy(sample)

    def get_dim_reduce_transform(self, data_transform: transforms, model_path: Path) -> transforms:
        # Only a client-specific digit is accepted as the condition for dimensionality reduction.
        assert self.condition.isdigit()
        self.load_autoencoder(model_path)
        return transforms.Compose([data_transform, self.dim_reduction_transform])

    def get_AE_loss(
        self,
        base_loss: _Loss,
        latent_dim: int,
    ) -> _Loss:
        return VAE_loss(latent_dim, base_loss)
