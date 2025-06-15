import torch

from fl4health.model_bases.feature_extractor_buffer import FeatureExtractorBuffer
from tests.test_utils.models_for_test import HierarchicalCnn


MODEL = HierarchicalCnn()


def test_feature_extractor_buffer_for_train_and_eval_mode() -> None:
    MODEL.train()
    buffer = FeatureExtractorBuffer(MODEL, {"h1_layer2": True})
    buffer._maybe_register_hooks()
    # Enable accumulating features
    buffer.enable_accumulating_features()

    # Batch size of 4
    input_tensor = torch.randn(4, 1, 16, 16)
    MODEL(input_tensor)
    # Buffer should have one feature
    assert len(buffer.extracted_features_buffers["h1_layer2"]) == 1
    assert buffer.extracted_features_buffers["h1_layer2"][0].shape == torch.Size([4, 16, 1, 1])
    # As features should be flattened, extracted features should have shape [4, 16]
    assert buffer.get_extracted_features()["h1_layer2"].shape == torch.Size([4, 16])

    input_tensor = torch.randn(4, 1, 16, 16)
    MODEL(input_tensor)
    # As accumulating features is enabled, the buffer should have two features
    assert len(buffer.extracted_features_buffers["h1_layer2"]) == 2
    # Features of each input batch saved as a separate tensor
    assert buffer.extracted_features_buffers["h1_layer2"][1].shape == torch.Size([4, 16, 1, 1])
    # As features should be flattened, final extracted features should have shape [8, 16]
    assert buffer.get_extracted_features()["h1_layer2"].shape == torch.Size([8, 16])

    # Disable accumulating features
    buffer.disable_accumulating_features()

    input_tensor = torch.randn(4, 1, 16, 16)
    MODEL(input_tensor)
    # As accumulating features is disabled, the buffer should have only one feature
    assert len(buffer.extracted_features_buffers["h1_layer2"]) == 1
    assert buffer.extracted_features_buffers["h1_layer2"][0].shape == torch.Size([4, 16, 1, 1])
    assert buffer.get_extracted_features()["h1_layer2"].shape == torch.Size([4, 16])

    buffer.remove_hooks()
    input_tensor = torch.randn(4, 1, 16, 16)
    MODEL(input_tensor)
    # As hooks are removed, the model should not add anything to the buffer
    assert len(buffer.extracted_features_buffers["h1_layer2"]) == 1

    buffer.clear_buffers()
    # As we cleared the buffer, we should have no features
    assert len(buffer.extracted_features_buffers["h1_layer2"]) == 0

    input_tensor = torch.randn(4, 1, 16, 16)
    MODEL(input_tensor)
    # As hooks are removed, the model should not add anything to the buffer
    assert len(buffer.extracted_features_buffers["h1_layer2"]) == 0

    buffer._maybe_register_hooks()
    input_tensor = torch.randn(4, 1, 16, 16)
    MODEL(input_tensor)
    input_tensor = torch.randn(4, 1, 16, 16)
    MODEL(input_tensor)
    # As accumulating features is disabled, the buffer should only have one feature
    assert len(buffer.extracted_features_buffers["h1_layer2"]) == 1

    buffer.enable_accumulating_features()
    input_tensor = torch.randn(4, 1, 16, 16)
    MODEL(input_tensor)
    input_tensor = torch.randn(4, 1, 16, 16)
    MODEL(input_tensor)
    # As accumulating features is enabled, the buffer should have three features
    assert len(buffer.extracted_features_buffers["h1_layer2"]) == 3

    # Put the model in eval mode
    MODEL.eval()
    buffer.clear_buffers()
    input_tensor = torch.randn(4, 1, 16, 16)
    MODEL(input_tensor)
    input_tensor = torch.randn(4, 1, 16, 16)
    MODEL(input_tensor)
    # The model in eval mode should still add features to the buffer
    assert len(buffer.extracted_features_buffers["h1_layer2"]) == 2
    assert buffer.extracted_features_buffers["h1_layer2"][1].shape == torch.Size([4, 16, 1, 1])
    # As features should be flattened, final extracted features should have shape [8, 16]
    assert buffer.get_extracted_features()["h1_layer2"].shape == torch.Size([8, 16])


def test_feature_extractor_buffer_with_hierarchical_layer_names() -> None:
    MODEL.train()
    feature_extraction_layers = {"h1_layer2": False, "h1_layer2.h2_layer2": False, "h1_layer2.h2_layer2.pool": False}
    # By default, accumulating features is disabled
    buffer = FeatureExtractorBuffer(MODEL, feature_extraction_layers)
    buffer._maybe_register_hooks()

    input_tensor = torch.randn(4, 1, 16, 16)
    MODEL(input_tensor)
    keys = list(buffer.extracted_features_buffers.keys())
    for i in range(len(keys)):
        assert len(buffer.extracted_features_buffers[keys[i]]) == 1
        # From generic to most specific layer, extracted features should be the same
        assert torch.all(buffer.get_extracted_features()[keys[i - 1]] == buffer.get_extracted_features()[keys[i]])

    MODEL.eval()
    input_tensor = torch.randn(4, 1, 16, 16)
    MODEL(input_tensor)
    for i in range(len(keys)):
        assert len(buffer.extracted_features_buffers[keys[i]]) == 1
        # From generic to most specific layer, extracted features should be the same
        assert torch.all(buffer.get_extracted_features()[keys[i - 1]] == buffer.get_extracted_features()[keys[i]])
