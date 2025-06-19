from flwr.common.typing import Scalar

from fl4health.checkpointing.checkpointer import FunctionTorchModuleCheckpointer


def score_function(_: float, metrics: dict[str, Scalar]) -> float:
    accuracy = metrics["accuracy"]
    precision = metrics["precision"]
    assert isinstance(accuracy, float)
    assert isinstance(precision, float)
    return 0.5 * (accuracy + precision)


def test_function_checkpointer() -> None:
    function_checkpointer = FunctionTorchModuleCheckpointer("", "", score_function, maximize=True)
    loss_1, loss_2 = 1.0, 0.9
    metrics_1: dict[str, Scalar] = {"accuracy": 0.87, "precision": 0.67, "f1": 0.76}
    metrics_2: dict[str, Scalar] = {"accuracy": 0.87, "precision": 0.9, "f1": 0.6}

    function_checkpointer.best_score = 0.85
    # Should be false since the best score seen is set to 0.85 above
    assert not function_checkpointer._should_checkpoint(score_function(loss_1, metrics_1))
    # Should be true since the average of accuracy and precision provided in the dictionary is larger than 0.85
    assert function_checkpointer._should_checkpoint(score_function(loss_2, metrics_2))
