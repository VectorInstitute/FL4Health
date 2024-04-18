from fl4health.checkpointing.checkpointer import BestMetricTorchCheckpointer


def test_best_metric_checkpointer() -> None:
    max_checkpointer = BestMetricTorchCheckpointer("", "", True)
    # First checkpoint should happen since the best metric is None
    none_checkpoint = max_checkpointer._should_checkpoint(0.95)
    assert none_checkpoint
    max_checkpointer.best_metric = 0.95

    # Second checkpoint shouldn't happen since it's not larger
    smaller_metric_checkpoint = max_checkpointer._should_checkpoint(0.92)
    assert not smaller_metric_checkpoint

    # third checkpoint should happen as it is larger
    larger_metric_checkpoint = max_checkpointer._should_checkpoint(0.96)
    assert larger_metric_checkpoint

    min_checkpointer = BestMetricTorchCheckpointer("", "", False)
    # First checkpoint should happen since the best metric is None
    none_checkpoint = min_checkpointer._should_checkpoint(0.15)
    assert none_checkpoint
    min_checkpointer.best_metric = 0.15

    # Second checkpoint should happen since it's smaller
    smaller_metric_checkpoint = min_checkpointer._should_checkpoint(0.12)
    assert smaller_metric_checkpoint

    # third checkpoint shouldn't happen as it is larger
    larger_metric_checkpoint = min_checkpointer._should_checkpoint(0.16)
    assert not larger_metric_checkpoint
