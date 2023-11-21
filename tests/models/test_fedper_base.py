from fl4health.model_bases.fedper_base import FedPerModel
from tests.test_utils.models_for_test import FeatureCnn, HeadCnn


def test_apfl_model_gets_correct_layers() -> None:
    model = FedPerModel(FeatureCnn(), HeadCnn())
    layers_to_exchange = model.layers_to_exchange()
    filtered_layer_names = [
        layer_name for layer_name in model.state_dict().keys() if layer_name.startswith("global_feature_extractor.")
    ]
    for test_layer, expected_layer in zip(layers_to_exchange, filtered_layer_names):
        assert test_layer == expected_layer
