from fl4health.model_bases.apfl_base import ApflModule
from tests.test_utils.models_for_test import ToyConvNet


def test_apfl_model_gets_correct_layers() -> None:
    model = ApflModule(ToyConvNet())
    layers_to_exchange = model.layers_to_exchange()
    filtered_layer_names = [layer_name for layer_name in model.state_dict() if layer_name.startswith("global_model.")]
    for test_layer, expected_layer in zip(layers_to_exchange, filtered_layer_names):
        assert test_layer == expected_layer
