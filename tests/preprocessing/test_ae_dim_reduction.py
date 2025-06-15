from pathlib import Path

import torch
from torch.utils.data import DataLoader

from fl4health.model_bases.autoencoders_base import ConditionalVae, VariationalAe
from fl4health.preprocessing.autoencoders.dim_reduction import CvaeVariableConditionProcessor, VaeProcessor
from fl4health.utils.dataset import BaseDataset
from tests.test_utils.models_for_test import VariationalDecoder, VariationalEncoder


PATH = Path("tests/utils/resources/autoencoder.pt")


class ConditionalDataset(BaseDataset):
    def __init__(self, data_size: int = 50, sample_vector_size: int = 10, condition_vector_size: int = 8) -> None:
        # 100 is the number of samples
        self.data = torch.randn(data_size, sample_vector_size)
        self.conditions = torch.rand(data_size, condition_vector_size)
        # "label" converter treats the target as the condition, therefore we set the target to our condition.
        self.targets = self.conditions

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.data[index], self.targets[index]


class SelfSupervisedTrainingDataset(BaseDataset):
    def __init__(self, data_size: int = 50, sample_vector_size: int = 10) -> None:
        # 100 is the number of samples
        self.data = torch.randn(data_size, sample_vector_size)
        self.targets = self.data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.data[index], self.targets[index]


def test_non_fixed_batched_conditional_dim_reduction() -> None:
    # Data settings
    data_size = 50
    sample_vector_size = 100
    condition_vector_size = 8
    batch_size = 10
    # Create a dummy dataset for testing
    dummy_dataset = ConditionalDataset(data_size, sample_vector_size, condition_vector_size)
    # We don't need to convert the data since we are not training a CVAE.
    data_loader = DataLoader(dummy_dataset, batch_size=batch_size)

    # CVAE Model Settings
    embedding_size = 2
    # Initiate Conditional Variational model
    # To be able to use these models, sample_vector_size should be 100.
    encoder = VariationalEncoder(embedding_size, condition_vector_size)
    decoder = VariationalDecoder(embedding_size, condition_vector_size)
    # unpack_input_condition can be none since we just use encoder.
    autoencoder = ConditionalVae(encoder=encoder, decoder=decoder)
    torch.save(autoencoder, PATH)
    # Initiating a non-fixed conditional processor as for each data sample we have a random condition vector.
    cvae_processor = CvaeVariableConditionProcessor(checkpointing_path=PATH)
    # Get a batch of data
    data_batch, condition_batch = next(iter(data_loader))
    encoded_batch = cvae_processor(data_batch, condition_batch)
    # Check the type and dimension of output: assuming data is "batch first".
    assert isinstance(encoded_batch, torch.Tensor) and encoded_batch.dim() == 2
    # Check the shape of encoder output
    assert encoded_batch.shape[0] == batch_size and encoded_batch.shape[1] == embedding_size * 2


def test_vae_dim_reduction() -> None:
    """Tests the VAE dimensionality reduction both on a single data and a batch of data."""
    # Data settings
    data_size = 50
    sample_vector_size = 100
    batch_size = 10
    # Create a dummy dataset for testing
    dummy_dataset = SelfSupervisedTrainingDataset(data_size, sample_vector_size)
    # We don't need to convert the data since we are not training a CVAE.
    data_loader = DataLoader(dummy_dataset, batch_size=batch_size)
    # VAE model
    embedding_size = 2
    encoder = VariationalEncoder(embedding_size)
    decoder = VariationalDecoder(embedding_size)
    autoencoder = VariationalAe(encoder=encoder, decoder=decoder)

    torch.save(autoencoder, PATH)
    # Initiating a dimensionality reduction preprocessor.
    vae_processor = VaeProcessor(checkpointing_path=PATH)
    # Get a batch of data
    data_batch, target_batch = next(iter(data_loader))
    encoded_batch = vae_processor(data_batch)
    # Check the type and dimension of output: assuming data is "batch first".
    assert isinstance(encoded_batch, torch.Tensor) and encoded_batch.dim() == 2
    # Check the shape of encoder output
    assert encoded_batch.shape[0] == batch_size and encoded_batch.shape[1] == embedding_size * 2

    # Trying one data sample
    sample, target = dummy_dataset[0]
    encoded_sample = vae_processor(sample)
    assert encoded_sample.size(0) == embedding_size * 2
