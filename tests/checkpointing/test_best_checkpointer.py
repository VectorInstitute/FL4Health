from fl4health.checkpointing.checkpointer import BestLossTorchCheckpointer


def test_best_metric_checkpointer() -> None:
    best_loss_checkpointer = BestLossTorchCheckpointer("", "")
    # First checkpoint should happen since the best metric is None
    none_checkpoint = best_loss_checkpointer._should_checkpoint(0.95)
    assert none_checkpoint
    best_loss_checkpointer.best_score = 0.95

    # Second checkpoint shouldn't happen since it's not smaller
    larger_metric_checkpoint = best_loss_checkpointer._should_checkpoint(1.1)
    assert not larger_metric_checkpoint

    # third checkpoint should happen as it is smaller
    smaller_metric_checkpoint = best_loss_checkpointer._should_checkpoint(0.87)
    assert smaller_metric_checkpoint
