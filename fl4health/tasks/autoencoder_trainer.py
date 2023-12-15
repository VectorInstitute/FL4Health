from pathlib import Path
from typing import Callable, Dict, Tuple

import torchvision.transforms as transforms
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader

from fl4health.processing.VAE_dim_reduction import ClientConditionedProcessor, VAEProcessor
from fl4health.processing.VAE_training import AETransformer
from fl4health.tasks.loss import VAE_loss
from fl4health.utils.sampler import LabelBasedSampler


class VAETrainer:
    """This Trainer class can enhance any client class within this framework by adding the functionality of training
    a Variational Auto-encoder. Functionalities include prepare_input()
    which calls the data loader with an specific VAE transformer for VAE training,
    and reduce_dim() which is used to reduce the dimension of data during the pre-processing step with VAEs.
    """

    def prepare_input(
        self, load_data: Callable, data_path: Path, batch_size: int, sampler: LabelBasedSampler, transform: transforms
    ) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
        """Calls the data loading function with the given parameters and adds a transofrmer
        to convert the data in the suitable format for VAE training i.e. (sample, sample) format.

        Args:
            load_data (Callable): Function to load the data.
            data_path (Path): Path to the data.
            batch_size (int): Size of the batches.
            sampler (LabelBasedSampler): Sampler for the data.
            transform (transforms): Transformation for the data samples.

        Returns:
            Tuple[DataLoader, DataLoader, Dict[str, int]]:
            Train and validation data loaders and additional information.
        """
        train_loader, val_loader, _ = load_data(
            data_dir=data_path,
            batch_size=batch_size,
            sampler=sampler,
            transform=transform,
            data_target_transform=AETransformer(),
        )
        return train_loader, val_loader, _

    def reduce_dim(
        self,
        model_path: Path,
        load_data: Callable,
        data_path: Path,
        batch_size: int,
        sampler: LabelBasedSampler,
        transform: transforms,
    ) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
        """Reduces data dimension using a trained VAE model.

        Args:
            model_path (Path): Path to the trained VAE model.
            load_data (Callable): Function to load the data.
            data_path (Path): Path to the data.
            batch_size (int): Size of the batches.
            sampler (LabelBasedSampler): Sampler for the data.
            transform (transforms): Transformation for the data samples.

        Returns:
            Tuple[DataLoader, DataLoader, Dict[str, int]]:
            Train and validation data loaders and additional information.
        """
        dim_reduction_processing = VAEProcessor(model_path)
        # dimensionality reduction is applied in the form of a data transformer on only the data samples
        transform = transforms.Compose([transform, dim_reduction_processing])
        train_loader, val_loader, _ = load_data(
            data_dir=data_path, batch_size=batch_size, sampler=sampler, transform=transform
        )
        return train_loader, val_loader, _

    def loss(
        self,
        latent_dim: int,
        base_loss: _Loss,
    ) -> _Loss:
        """Calculates the loss function for the VAE.

        Args:
            latent_dim (int): Dimensionality of the latent space.
            base_loss (_Loss): Base loss function defined by the user.

        Returns:
            _Loss: Calculated loss function for the VAE.
        """
        return VAE_loss(latent_dim, base_loss)


class CVAETrainer:
    """This Trainer class adds the functionality of training a Conditional Variational Auto-encoder
    to any client class within this framework. Functionalities include prepare_input() which calls
    the data loader with an specific conidtional VAE transformer to enable the CVAE training,
    and reduce_dim() which enables conditional dimensionality reduction pre-processing with CVAEs.
    """

    def __init__(self, condition: str = "label"):
        self.condition = condition

    def prepare_input(
        self, load_data: Callable, data_path: Path, batch_size: int, sampler: LabelBasedSampler, transform: transforms
    ) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
        """Calls the data loading function with the given parameters and adds a transofrmer
        to convert the data in the suitable format for CVAE training i.e. ([sample,condition], sample).

        Args:
            load_data (Callable): Function to load the data.
            data_path (Path): Path to the data.
            batch_size (int): Size of the batches.
            sampler (LabelBasedSampler): Sampler for the data.
            transform (transforms): Transformation for the data samples.

        Returns:
            Tuple[DataLoader, DataLoader, Dict[str, int]]:
            Train and validation data loaders and additional information.
        """
        train_loader, val_loader, _ = load_data(
            data_dir=data_path,
            batch_size=batch_size,
            sampler=sampler,
            transform=transform,
            data_target_transform=AETransformer(self.condition),
        )
        return train_loader, val_loader, _

    def reduce_dim(
        self,
        model_path: Path,
        load_data: Callable,
        data_path: Path,
        batch_size: int,
        sampler: LabelBasedSampler,
        transform: transforms,
    ) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
        """Reduces data dimension using a trained CVAE model.

        Args:
            model_path (Path): Path to the trained CVAE model.
            load_data (Callable): Function to load the data.
            data_path (Path): Path to the data.
            batch_size (int): Size of the batches.
            sampler (LabelBasedSampler): Sampler for the data.
            transform (transforms): Transformation for the data samples.

        Returns:
            Tuple[DataLoader, DataLoader, Dict[str, int]]:
            Train and validation data loaders and additional information.
        """

        # Condition on a client-specific digit
        assert self.condition.isdigit()
        dim_reduction_processing = ClientConditionedProcessor(model_path, int(self.condition))
        transform = transforms.Compose([transform, dim_reduction_processing])

        train_loader, val_loader, _ = load_data(
            data_dir=data_path,
            batch_size=batch_size,
            sampler=sampler,
            transform=transform,
        )
        return train_loader, val_loader, _

    def loss(
        self,
        latent_dim: int,
        base_loss: _Loss,
    ) -> _Loss:
        """Calculates the loss function for the CVAE.

        Args:
            latent_dim (int): Dimensionality of the latent space.
            base_loss (_Loss): Base loss function defined by the user.

        Returns:
            _Loss: Calculated loss function for CVAE.
        """
        return VAE_loss(latent_dim, base_loss)
