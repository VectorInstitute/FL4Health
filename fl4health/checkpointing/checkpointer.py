import os
from logging import INFO
from typing import Optional

import torch
import torch.nn as nn
from flwr.common.logger import log


class TorchCheckpointer:
    def __init__(self, best_checkpoint_dir: str, best_checkpoint_name: str) -> None:
        self.best_checkpoint_path = os.path.join(best_checkpoint_dir, best_checkpoint_name)

    def maybe_checkpoint(self, model: nn.Module, _: Optional[float] = None) -> None:
        raise NotImplementedError

    def load_best_checkpoint(self) -> nn.Module:
        return torch.load(self.best_checkpoint_path)


class LatestTorchCheckpointer(TorchCheckpointer):
    def __init__(self, best_checkpoint_dir: str, best_checkpoint_name: str) -> None:
        super().__init__(best_checkpoint_dir, best_checkpoint_name)

    def maybe_checkpoint(self, model: nn.Module, _: Optional[float] = None) -> None:
        # Always checkpoint the latest model
        log(INFO, "Saving latest checkpoint with LatestTorchCheckpointer")
        torch.save(model, self.best_checkpoint_path)


class BestMetricTorchCheckpointer(TorchCheckpointer):
    def __init__(self, best_checkpoint_dir: str, best_checkpoint_name: str, maximize: bool = False) -> None:
        super().__init__(best_checkpoint_dir, best_checkpoint_name)
        self.best_metric: Optional[float] = None
        # Whether we're looking to maximize the metric (alternatively minimize)
        self.maximize = maximize
        self.comparison_str = ">=" if self.maximize else "<="

    def _should_checkpoint(self, comparison_metric: float) -> bool:
        # Compares the current metric to the best previously recorded, returns true if should checkpoint and false
        # otherwise
        if self.best_metric:
            if self.maximize:
                return self.best_metric <= comparison_metric
            else:
                return self.best_metric >= comparison_metric

        # If best metric is none, then this is the first checkpoint
        return True

    def maybe_checkpoint(self, model: nn.Module, comparison_metric: Optional[float] = None) -> None:
        assert comparison_metric is not None
        if self._should_checkpoint(comparison_metric):
            log(
                INFO,
                f"Checkpointing the model: Current metric ({comparison_metric}) "
                f"{self.comparison_str} Best metric ({self.best_metric})",
            )
            self.best_metric = comparison_metric
            torch.save(model, self.best_checkpoint_path)
        else:
            log(
                INFO,
                f"Not checkpointing the model: Current metric ({comparison_metric}) is not "
                f"{self.comparison_str} Best metric ({self.best_metric})",
            )
