import torch
from torch.utils.data import DataLoader

from fl4health.utils.dataset import TensorDataset
from fl4health.utils.dataset_converter import AutoEncoderDatasetConverter, DatasetConverter


def get_dummy_dataset() -> TensorDataset:
    data = torch.randn(100, 10, 8)
    targets = torch.randint(5, (100,))
    return TensorDataset(data=data, targets=targets)


def test_dataset_converter() -> None:
    dummy_dataset = get_dummy_dataset()

    # Create a dummy converter function for testing
    def dummy_converter(data: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return data, target

    # Test DatasetConverter
    dataset_converter = DatasetConverter(dummy_converter, dummy_dataset)
    # Test __getitem__
    sample = dataset_converter[0]
    assert isinstance(sample, tuple) and len(sample) == 2
    # Test __len__
    assert len(dataset_converter) == len(dummy_dataset)
    # Test convert_dataset
    new_dataset = dataset_converter.convert_dataset(dummy_dataset)
    assert new_dataset is dataset_converter
    assert dataset_converter.dataset is dummy_dataset


def test_autoencoder_converter_tensor_conditioned() -> None:
    # Create a dummy dataset for testing
    dummy_dataset = get_dummy_dataset()
    # Test AutoEncoderDatasetConverter with a vector condition.
    condition_tensor = torch.Tensor([4, 0, 5, 3])
    autoencoder_converter = AutoEncoderDatasetConverter(condition=condition_tensor)
    autoencoder_converter.convert_dataset(dummy_dataset)
    # Test __getitem__
    sample = autoencoder_converter[0]
    assert isinstance(sample, tuple) and len(sample) == 2
    # Test __len__
    assert len(autoencoder_converter) == len(dummy_dataset)
    # Test get_condition_vector_size
    condition_size = autoencoder_converter.get_condition_vector_size()
    assert condition_size == condition_tensor.shape[0]
    # Test _only_replace_target_with_data converter function
    data, target = torch.randn(10, 8), torch.randint(5, (1,))
    result_data, result_target = autoencoder_converter._only_replace_target_with_data(data, target)
    # Data and Target should have equal size.
    assert torch.equal(result_data, data) and torch.equal(result_target, data)

    # Test _cat_input_condition converter function
    result_data, result_target = autoencoder_converter._cat_input_condition(data, target)
    assert (result_data.shape[0] == data.shape[0] * data.shape[1] + condition_size) and torch.equal(
        result_target, data
    )


def test_autoencoder_converter_label_conditioned() -> None:
    # Create a dummy dataset for testing
    dummy_dataset = get_dummy_dataset()

    assert dummy_dataset.targets is not None

    # Initialize the converter and convert the dataset.
    autoencoder_converter_label = AutoEncoderDatasetConverter(condition="label", do_one_hot_encoding=True)
    autoencoder_converter_label.convert_dataset(dummy_dataset)
    # Get the size of condition vector (condition is one_hot_encoded target)
    num_conditions = autoencoder_converter_label.get_condition_vector_size()
    # Get an original data sample from the dummy dataset.
    original_data, _ = dummy_dataset.data[0], dummy_dataset.targets[0]
    # Get the converted version of that.
    converted_data, converted_target = autoencoder_converter_label[0]
    assert (
        converted_data.shape[0] == original_data.shape[0] * original_data.shape[1] + num_conditions
    ) and torch.equal(converted_target, original_data)


def test_autoencoder_converter_with_custom_conversion_function() -> None:
    # Create a dummy dataset for testing
    dummy_dataset = get_dummy_dataset()

    assert dummy_dataset.targets is not None

    # Initialize the converter and convert the dataset.
    autoencoder_converter_label = AutoEncoderDatasetConverter(condition="label", do_one_hot_encoding=True)
    autoencoder_converter_label.convert_dataset(dummy_dataset)

    autoencoder_converter_custom_conversion_function = AutoEncoderDatasetConverter(
        None, False, autoencoder_converter_label._cat_input_label, 5
    )
    autoencoder_converter_custom_conversion_function.convert_dataset(dummy_dataset)

    # Get the size of condition vector (condition is one_hot_encoded target)
    target_num_conditions = autoencoder_converter_label.get_condition_vector_size()
    num_conditions = autoencoder_converter_custom_conversion_function.get_condition_vector_size()
    assert target_num_conditions == num_conditions

    # Because we're using the same conversion function ( one as a custom function, these should be the same)
    target_converted_data, target_converted_target = autoencoder_converter_label[0]
    converted_data, converted_target = autoencoder_converter_custom_conversion_function[0]
    assert torch.equal(converted_target, target_converted_target)
    assert torch.equal(converted_data, target_converted_data)


def test_pack_unpack() -> None:
    batch_size = 10
    # Create a dummy dataset for testing
    dummy_dataset = get_dummy_dataset()

    assert dummy_dataset.targets is not None

    # Initiate the data converter
    autoencoder_converter = AutoEncoderDatasetConverter(condition="label", do_one_hot_encoding=True)
    # Convert the dataset
    converted_data = autoencoder_converter.convert_dataset(dummy_dataset)
    # create data loader
    data_loader = DataLoader(converted_data, batch_size=batch_size)
    # A normal training loop
    unpacking_function = autoencoder_converter.get_unpacking_function()
    for all_data_batch, target_batch in data_loader:
        data_batch, condition_batch = unpacking_function(all_data_batch)
        # Check the unpacked data shape is as expected.
        assert data_batch.shape == torch.Size(
            [batch_size, autoencoder_converter.data_shape[0], autoencoder_converter.data_shape[1]]
        )
        # Check the unpacked condition shape is as expected.
        assert condition_batch.shape == torch.Size([batch_size, autoencoder_converter.get_condition_vector_size()])
        # Check the target shape is the same as input shape after unpacking.
        assert target_batch.shape == data_batch.shape
