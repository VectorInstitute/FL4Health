from unittest.mock import patch

import pytest
import torch
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader

from fl4health.clients.gpfl_client import GpflClient
from fl4health.model_bases.gpfl_base import Gce
from fl4health.parameter_exchange.layer_exchanger import FixedLayerExchanger
from fl4health.utils.dataset import TensorDataset
from tests.clients.fixtures import get_gpfl_client  # noqa
from tests.test_utils.models_for_test import FeatureCnn, HeadCnn


@pytest.mark.parametrize("global_module,head_module,feature_dim,num_classes", [(FeatureCnn(), HeadCnn(), 256, 32)])
def test_gpfl_set_optimizer(get_gpfl_client: GpflClient) -> None:  # noqa
    """Test that GpflClient checks that optimizer dictionary is valid."""
    client = get_gpfl_client
    with patch.object(client, "get_optimizer") as mock_get_optimizer:
        # Mock the get_optimizer method to return an invalid dictionary
        mock_get_optimizer.return_value = {"model": get_gpfl_client.model.parameters()}
        # Assert that set_optimizer raises AssertionError
        with pytest.raises(AssertionError):
            client.set_optimizer({})


@pytest.mark.parametrize("global_module,head_module,feature_dim,num_classes", [(FeatureCnn(), HeadCnn(), 256, 32)])
def test_gpfl_model_type(get_gpfl_client: GpflClient) -> None:  # noqa
    """Test that the correct model base is being used."""
    client = get_gpfl_client
    assert isinstance(client.get_parameter_exchanger({}), FixedLayerExchanger)
    # Change client's model, ignoring type as we are testing typing assertion
    client.model = HeadCnn()  # type: ignore
    with pytest.raises(AssertionError):
        client.get_parameter_exchanger({})


@pytest.mark.parametrize("global_module,head_module,feature_dim,num_classes", [(FeatureCnn(), HeadCnn(), 256, 2)])
def test_gpfl_setup_client(get_gpfl_client: GpflClient) -> None:  # noqa
    """Test that the initial parameters are set and calculated correctly."""
    torch.manual_seed(42)
    client = get_gpfl_client
    basic_client = type(client).__bases__[0]
    data_samples = 10
    batch_size = 2
    with patch.object(basic_client, "setup_client") as mock_super_setup_client:
        mock_super_setup_client.return_value = None
        data_samples = 10
        batch_size = 2
        dummy_data = torch.randn(data_samples, 1, 28, 28)
        dummy_labels = torch.randint(0, 2, (data_samples,))
        train_dataset = TensorDataset(dummy_data, dummy_labels)
        client.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        client.setup_client({})

        assert client.num_classes == 2
        assert client.feature_dim == 256
        assert torch.equal(client.class_sample_proportion, torch.Tensor([0.5, 0.5]))

        # Test with one-hot encoded labels
        dummy_one_hot_labels = one_hot(dummy_labels, num_classes=2)
        print(dummy_one_hot_labels)
        one_hot_train_dataset = TensorDataset(dummy_data, dummy_one_hot_labels)
        client.train_loader = DataLoader(one_hot_train_dataset, batch_size=batch_size, shuffle=True)
        client.setup_client({})
        assert client.num_classes == 2
        assert client.feature_dim == 256
        assert torch.equal(client.class_sample_proportion, torch.Tensor([0.5, 0.5]))

    torch.seed()  # resetting the seed at the end, just to be safe


@pytest.mark.parametrize("global_module,head_module,feature_dim,num_classes", [(FeatureCnn(), HeadCnn(), 4, 3)])
def test_compute_conditional_inputs(get_gpfl_client: GpflClient) -> None:  # noqa
    """Given a frozen GCE, conditional inputs should be computed correctly."""
    client = get_gpfl_client
    # Create a mock GCE embedding of shape (num_classes, feature_dim)
    manual_embedding = torch.Tensor([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.9, 1.0, 1.1, 1.2]])
    manual_class_sample_proportion = torch.Tensor([0.3, 0.4, 0.3])
    # Just assigning a dummy GCE model since we don't need it for this test.
    client.gce_frozen = Gce(4, 3)
    client.feature_dim = 4
    client.num_classes = 3
    client.gce_frozen.embedding.weight.data = manual_embedding
    client.class_sample_proportion = manual_class_sample_proportion
    # Manually compute the conditional inputs
    manual_g = (manual_embedding[0, :] + manual_embedding[1, :] + manual_embedding[2, :]) / 3
    manual_p = (
        manual_embedding[0, :] * manual_class_sample_proportion[0]
        + manual_embedding[1, :] * manual_class_sample_proportion[1]
        + manual_embedding[2, :] * manual_class_sample_proportion[2]
    )
    manual_p = manual_p / client.num_classes
    # Call compute_conditional_inputs in client to calculate p and g
    client.compute_conditional_inputs()
    assert torch.allclose(client.personalized_conditional_input, manual_p, atol=1e-6), (
        "Computed personalized condition does not match the manual calculation."
    )
    assert torch.allclose(client.global_conditional_input, manual_g, atol=1e-6), (
        "Computed global condition does not match the manual calculation."
    )


@pytest.mark.parametrize("global_module,head_module,feature_dim,num_classes", [(FeatureCnn(), HeadCnn(), 256, 32)])
def test_transform_input_and_predict(get_gpfl_client: GpflClient) -> None:  # noqa
    """Test that ``transform_input`` method works as expected, and right predictions are returned."""
    client = get_gpfl_client
    batch_size = 8
    input = torch.randn(batch_size, 1, 28, 28)
    # Make sure that conditional inputs are attached to the input data when ``transform_input`` is called.
    global_conditional_input = torch.ones(256)
    personalized_conditional_input = torch.ones(256)
    client.global_conditional_input = global_conditional_input
    client.personalized_conditional_input = personalized_conditional_input
    transformed_input = client.transform_input(input)
    assert isinstance(transformed_input, dict)
    assert {"input", "global_conditional_input", "personalized_conditional_input"} == set(transformed_input.keys())
    assert torch.equal(transformed_input["global_conditional_input"], global_conditional_input)
    assert torch.equal(transformed_input["personalized_conditional_input"], personalized_conditional_input)

    # Test the model's forward method to return dummy predictions and features
    client.model.train()
    preds, features = client.predict(transformed_input)
    # Assert the model prediction and features shapes.
    assert preds["prediction"].shape == (batch_size, 32)
    assert {"local_features", "global_features"} == set(features.keys())
    assert features["local_features"].shape == features["global_features"].shape == (batch_size, 256)

    # Test predict function for evaluation
    client.model.eval()
    preds, features = client.predict(transformed_input)
    # Assert the model prediction and features shapes for evaluation
    assert preds["prediction"].shape == (batch_size, 32)
    assert "local_features" not in features
    assert "global_features" not in features


@pytest.mark.parametrize("global_module,head_module,feature_dim,num_classes", [(FeatureCnn(), HeadCnn(), 256, 32)])
def test_gpfl_training_loss(get_gpfl_client: GpflClient) -> None:  # noqa
    """Test the loss computation in the GPFL client."""
    client = get_gpfl_client
    global_conditional_input = torch.ones(256)
    personalized_conditional_input = torch.ones(256)
    client.global_conditional_input = global_conditional_input
    client.personalized_conditional_input = personalized_conditional_input

    batch_size = 8
    input = torch.randn(batch_size, 1, 28, 28)
    target = torch.randint(0, 32, (batch_size,))
    client.criterion = torch.nn.CrossEntropyLoss()
    client.gce_frozen = client.model.gce
    transformed_input = client.transform_input(input)
    client.model.train()
    preds, features = client.predict(transformed_input)
    loss = client.compute_training_loss(preds, features, target)
    assert {"prediction_loss", "gce_softmax_loss", "magnitude_level_loss"} == set(loss.additional_losses.keys())
    assert torch.allclose(
        loss.backward["backward"],
        (
            loss.additional_losses["prediction_loss"]
            + loss.additional_losses["gce_softmax_loss"]
            + loss.additional_losses["magnitude_level_loss"] * client.lam
        ),
    )


@pytest.mark.parametrize("global_module,head_module,feature_dim,num_classes", [(FeatureCnn(), HeadCnn(), 256, 32)])
def test_gpfl_set_optimizer_l2(get_gpfl_client: GpflClient) -> None:  # noqa
    """Test that the optimizers and model L2 regularization are set correctly."""
    client = get_gpfl_client
    # Mock client set_optimizer
    with patch.object(client, "get_optimizer") as mock_get_optimizer:
        mock_get_optimizer.return_value = {
            "model": torch.optim.SGD(client.model.gpfl_main_module.parameters(), lr=0.01),
            "gce": torch.optim.SGD(client.model.gce.embedding.parameters(), lr=0.01),
            "cov": torch.optim.SGD(client.model.cov.parameters(), lr=0.01),
        }
        client.set_optimizer({})
        # Assert that the optimizers weight decay are now set correctly
        gce_weight_decay = client.optimizers["gce"].param_groups[0].get("weight_decay", 0.0)
        assert gce_weight_decay == client.mu
        cov_weight_decay = client.optimizers["cov"].param_groups[0].get("weight_decay", 0.0)
        assert cov_weight_decay == client.mu


@pytest.mark.parametrize("global_module,head_module,feature_dim,num_classes", [(FeatureCnn(), HeadCnn(), 256, 32)])
def test_calculate_class_sample_proportions(get_gpfl_client: GpflClient) -> None:  # noqa
    """Test the calculation of class sample proportions."""
    torch.manual_seed(64)
    client = get_gpfl_client
    data_samples = 100
    batch_size = 10
    num_classes = 3
    client.num_classes = num_classes
    dummy_data = torch.randn(data_samples, 1, 28, 28)
    dummy_labels = torch.randint(0, num_classes, (data_samples,))
    train_dataset = TensorDataset(dummy_data, dummy_labels)
    client.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # Assert that the class sample proportions are calculated correctly
    expected_proportions = torch.tensor([0.3100, 0.3600, 0.3300])
    assert torch.equal(client.calculate_class_sample_proportions(), expected_proportions)

    # Now test with one-hot encoded labels
    dummy_one_hot_labels = one_hot(dummy_labels, num_classes=num_classes)
    one_hot_train_dataset = TensorDataset(dummy_data, dummy_one_hot_labels)
    client.train_loader = DataLoader(one_hot_train_dataset, batch_size=batch_size, shuffle=True)
    assert torch.equal(client.calculate_class_sample_proportions(), expected_proportions)

    torch.seed()  # resetting the seed at the end, just to be safe
