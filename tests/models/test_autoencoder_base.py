import torch
from torch.utils.data import DataLoader

from fl4health.model_bases.autoencoders_base import ConditionalVae, VariationalAe
from fl4health.utils.dataset import TensorDataset
from fl4health.utils.dataset_converter import AutoEncoderDatasetConverter
from tests.test_utils.models_for_test import VariationalDecoder, VariationalEncoder


def get_dummy_dataset(data_size: int = 100) -> TensorDataset:
    data = torch.randn(100, data_size)
    targets: torch.Tensor = torch.randint(5, (data_size,))
    return TensorDataset(data=data, targets=targets)


def test_variational_autoencoder_model_base() -> None:
    # Model and data settings
    embedding_size = 2
    batch_size = 10
    data_vector_size = 100
    # Initiate Variational model
    encoder = VariationalEncoder(embedding_size)
    decoder = VariationalDecoder(embedding_size)
    autoencoder = VariationalAe(encoder=encoder, decoder=decoder)

    # Create a dummy dataset for testing
    dummy_dataset = get_dummy_dataset(data_vector_size)
    # Test AutoEncoderDatasetConverter with no condition.
    autoencoder_converter = AutoEncoderDatasetConverter(condition=None)
    converted_data = autoencoder_converter.convert_dataset(dummy_dataset)

    # Create data loader
    data_loader = DataLoader(converted_data, batch_size=batch_size)
    # Get a batch from the data loader and do a forward pass
    data, target = next(iter(data_loader))
    output = autoencoder(data)
    # Check the type and dimension of output: assuming data is "batch first".
    assert isinstance(output, torch.Tensor) and output.dim() == 2
    # Check the shape of output after concatenation in forward pass.
    assert output.shape[0] == batch_size and output.shape[1] == embedding_size * 2 + data_vector_size


def test_conditional_variational_autoencoder_model_base() -> None:
    # Data setting
    data_vector_size = 100
    # Create a dummy dataset for testing
    dummy_dataset = get_dummy_dataset(data_vector_size)
    # Test AutoEncoderDatasetConverter with condition based on data label.
    autoencoder_converter = AutoEncoderDatasetConverter(condition="label", do_one_hot_encoding=True)
    converted_data = autoencoder_converter.convert_dataset(dummy_dataset)

    # Model Settings
    embedding_size = 2
    batch_size = 10
    condition_vector_size = autoencoder_converter.get_condition_vector_size()
    # Initiate Conditional Variational model
    encoder = VariationalEncoder(embedding_size, condition_vector_size)
    decoder = VariationalDecoder(embedding_size, condition_vector_size)
    autoencoder = ConditionalVae(
        encoder=encoder,
        decoder=decoder,
    )
    autoencoder.unpack_input_condition = autoencoder_converter.get_unpacking_function()

    # Create data loader
    data_loader = DataLoader(converted_data, batch_size=batch_size)
    # Get a batch from the data loader and do a forward pass
    data, target = next(iter(data_loader))
    output = autoencoder(data)
    # Check the type and dimension of output: assuming data is "batch first".
    assert isinstance(output, torch.Tensor) and output.dim() == 2
    # Check the shape of output after concatenation in forward pass.
    assert output.shape[0] == batch_size and output.shape[1] == embedding_size * 2 + data_vector_size
