from fl4health.model_bases.fenda_base import FendaModel, FendaModelWithFeatureState
from tests.test_utils.models_for_test import FeatureCnn, FendaHeadCnn


def test_fenda_model_gets_correct_layers() -> None:
    model = FendaModel(FeatureCnn(), FeatureCnn(), FendaHeadCnn())
    layers_to_exchange = model.layers_to_exchange()
    filtered_layer_names = [
        layer_name for layer_name in model.state_dict() if layer_name.startswith("second_feature_extractor.")
    ]
    for test_layer, expected_layer in zip(layers_to_exchange, filtered_layer_names):
        assert test_layer == expected_layer


def test_fenda_model_with_feature_state_gets_correct_layers() -> None:
    model = FendaModelWithFeatureState(FeatureCnn(), FeatureCnn(), FendaHeadCnn())
    layers_to_exchange = model.layers_to_exchange()
    filtered_layer_names = [
        layer_name for layer_name in model.state_dict() if layer_name.startswith("second_feature_extractor.")
    ]
    for test_layer, expected_layer in zip(layers_to_exchange, filtered_layer_names):
        assert test_layer == expected_layer
