from fl4health.mixins.core_protocols import (
    FlexibleClientProtocol,
    FlexibleClientProtocolPreSetup,
    NumPyClientMinimalProtocol,
)


def test_numpy_client_protocol_attr() -> None:
    """Test interface for NumPyClientMinimalProtocol."""
    assert hasattr(NumPyClientMinimalProtocol, "get_parameters")
    assert hasattr(NumPyClientMinimalProtocol, "fit")
    assert hasattr(NumPyClientMinimalProtocol, "evaluate")
    assert hasattr(NumPyClientMinimalProtocol, "set_parameters")
    assert hasattr(NumPyClientMinimalProtocol, "update_after_train")


def test_flexible_client_presetup_protocol_attr() -> None:
    """Test interface for FlexibleClientProtocolPreSetup."""
    annotations = FlexibleClientProtocolPreSetup.__annotations__
    assert "device" in annotations
    assert "initialized" in annotations

    assert hasattr(FlexibleClientProtocolPreSetup, "setup_client")
    assert hasattr(FlexibleClientProtocolPreSetup, "get_model")
    assert hasattr(FlexibleClientProtocolPreSetup, "get_data_loaders")
    assert hasattr(FlexibleClientProtocolPreSetup, "get_optimizer")
    assert hasattr(FlexibleClientProtocolPreSetup, "get_criterion")
    assert hasattr(FlexibleClientProtocolPreSetup, "compute_loss_and_additional_losses")


def test_flexible_client_protocol_attr() -> None:
    """Test interface for FlexibleClientProtocol."""
    annotations = FlexibleClientProtocol.__annotations__
    assert "model" in annotations
    assert "optimizers" in annotations
    assert "train_loader" in annotations
    assert "val_loader" in annotations
    assert "test_loader" in annotations
    assert "criterion" in annotations

    assert hasattr(FlexibleClientProtocol, "initialize_all_model_weights")
    assert hasattr(FlexibleClientProtocol, "update_before_train")
    assert hasattr(FlexibleClientProtocol, "_compute_preds_and_losses")
    assert hasattr(FlexibleClientProtocol, "_apply_backwards_on_losses_and_take_step")
    assert hasattr(FlexibleClientProtocol, "_train_step_with_model_and_optimizer")
    assert hasattr(FlexibleClientProtocol, "_val_step_with_model")
    assert hasattr(FlexibleClientProtocol, "predict_with_model")
    assert hasattr(FlexibleClientProtocol, "transform_target")
    assert hasattr(FlexibleClientProtocol, "_transform_gradients_with_model")
    assert hasattr(FlexibleClientProtocol, "transform_gradients")
    assert hasattr(FlexibleClientProtocol, "compute_training_loss")
    assert hasattr(FlexibleClientProtocol, "validate")
    assert hasattr(FlexibleClientProtocol, "compute_evaluation_loss")
