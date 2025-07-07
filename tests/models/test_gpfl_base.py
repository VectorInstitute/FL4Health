import pytest
import torch

from fl4health.model_bases.gpfl_base import CoV, Gce, GpflModel
from tests.test_utils.models_for_test import FeatureCnn, HeadCnn


def test_gpfl_model_gets_correct_layers() -> None:
    gpfl_model = GpflModel(FeatureCnn(), HeadCnn(), feature_dim=16 * 4 * 4, num_classes=32)
    layers_to_exchange = gpfl_model.layers_to_exchange()
    base_layers = [
        layer_name for layer_name in gpfl_model.state_dict() if layer_name.startswith("gpfl_main_module.base_module.")
    ]
    head_layers = [
        layer_name for layer_name in gpfl_model.state_dict() if layer_name.startswith("gpfl_main_module.head_module.")
    ]
    cov_layers = [layer_name for layer_name in gpfl_model.state_dict() if layer_name.startswith("cov.")]
    gce_layers = [layer_name for layer_name in gpfl_model.state_dict() if layer_name.startswith("gce.")]
    for layer_name in base_layers:
        assert layer_name in layers_to_exchange
    for layer_name in head_layers:
        assert layer_name not in layers_to_exchange
    for layer_name in cov_layers:
        assert layer_name in layers_to_exchange
    for layer_name in gce_layers:
        assert layer_name in layers_to_exchange

    gpfl_layers_to_transfer = gpfl_model.layers_to_exchange()
    assert gpfl_layers_to_transfer == base_layers + cov_layers + gce_layers


def test_gpfl_model_forward_steps() -> None:
    gpfl_model = GpflModel(FeatureCnn(), HeadCnn(), feature_dim=16 * 4 * 4, num_classes=32)
    gpfl_model.train()
    input = torch.randn(8, 1, 28, 28)
    global_conditional_input = torch.randn(16 * 4 * 4)
    personalized_conditional_input = torch.randn(16 * 4 * 4)
    pred, features = gpfl_model(input, global_conditional_input, personalized_conditional_input)
    assert pred["prediction"].shape == (8, 32)
    assert "local_features" in features and "global_features" in features
    assert features["local_features"].shape == (8, 16 * 4 * 4)
    assert features["global_features"].shape == (8, 16 * 4 * 4)

    gpfl_model.eval()
    pred, features = gpfl_model(input, global_conditional_input, personalized_conditional_input)
    assert "prediction" in pred
    assert features == {}


def test_gpfl_conditional_input_shapes() -> None:
    gpfl_model = GpflModel(FeatureCnn(), HeadCnn(), feature_dim=16 * 4 * 4, num_classes=32)
    gpfl_model.train()
    input = torch.randn(8, 1, 28, 28)
    # global_conditional_input has a wrong shape
    global_conditional_input = torch.randn(16 * 1 * 1)
    personalized_conditional_input = torch.randn(16 * 4 * 4)
    with pytest.raises(AssertionError):
        pred, features = gpfl_model(input, global_conditional_input, personalized_conditional_input)


def test_cov_module() -> None:
    feature_dim = 16 * 4 * 4
    cov_module = CoV(feature_dim)
    # Check the structure of the cov model
    gamma_linear = cov_module.conditional_gamma[0]
    assert gamma_linear.in_features == feature_dim
    assert gamma_linear.out_features == feature_dim

    beta_linear = cov_module.conditional_beta[0]
    assert beta_linear.in_features == feature_dim
    assert beta_linear.out_features == feature_dim

    # Test the forward pass
    batch_size = 8
    feature_tensor = torch.randn(batch_size, feature_dim)
    context_tensor = torch.randn(batch_size, feature_dim)

    output = cov_module(feature_tensor, context_tensor)
    # Check the output shape and type
    assert isinstance(output, torch.Tensor)
    assert output.shape == (batch_size, feature_dim) == feature_tensor.shape

    # Manual expected output calculation
    expected_output = torch.relu(
        torch.mul(cov_module.conditional_gamma(context_tensor) + 1, feature_tensor)
        + cov_module.conditional_beta(context_tensor)
    )
    assert torch.allclose(output, expected_output, atol=1e-6)


def test_gce_module() -> None:
    feature_dim = 5
    num_classes = 2
    batch_size = 2
    gce_module = Gce(feature_dim, num_classes)
    # Check GCE module logic
    assert gce_module.embedding.weight.shape == (num_classes, feature_dim)
    # Set embedding weights to a 1.0 for testing
    gce_module.embedding.weight.data.fill_(1.0)
    # Compute the loss of a test tensor
    feature = torch.randn(batch_size, feature_dim)
    embeddings = gce_module.embedding.weight.data
    cosine_similarity = torch.zeros(batch_size, num_classes)
    for i, feat in enumerate(feature):
        for j, emb in enumerate(embeddings):
            cosine_similarity[i, j] = torch.dot(feat, emb) / (torch.norm(feat) * torch.norm(emb))

    label = torch.Tensor([0, 1]).reshape(-1)
    one_hot = torch.Tensor([[1, 0], [0, 1]]).reshape(-1, 2)
    expected_log_softmax_loss = torch.log_softmax(cosine_similarity, dim=1)
    expected_log_softmax_loss = -torch.mean(torch.sum(one_hot * expected_log_softmax_loss, dim=1))
    output = gce_module(feature, label)
    assert torch.allclose(output, expected_log_softmax_loss, atol=1e-6)

    # Test that the GCE module can also handle one-hot tensor labels
    assert one_hot.shape[1] == num_classes
    output = gce_module(feature, one_hot)
    assert torch.allclose(output, expected_log_softmax_loss, atol=1e-6)

    # Test the lookup functionality of GCE module
    lookedup_emb = gce_module.lookup(label)
    assert torch.equal(embeddings, lookedup_emb)
    # Now try one-hot encoded lookup
    lookedup_emb = gce_module.lookup(one_hot)
    assert torch.equal(embeddings, lookedup_emb)
