from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Optional, Union


class MetricType(Enum):
    TRAINING = "Training"
    VALIDATION = "Validation"


class MetricScope(Enum):
    PREDICTION = "prediction"
    PERSONAL = "personal"
    GLOBAL = "global"
    LOCAL = "local"


class LossType(Enum):
    LOSS = "loss"
    CHECKPOINT = "checkpoint"
    BACKWARD = "backward"
    PROXIMAL = "proximal_loss"
    GLOBAL = "global"
    LOCAL = "local"


class MetricChecker(ABC):
    def assert_metrics_exist(self, logs: str) -> None:
        log_lines = logs.split("\n")
        for log_line in log_lines:
            if self.should_check(log_line):
                self.check(log_line)

        assert self.found_all_metrics(), (
            f"Full output:\n{logs}\n" f"[ASSERT ERROR] Metrics {self.to_dict()} not found in logs."
        )

    @abstractmethod
    def should_check(self, log_line: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def check(self, log_line: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def found_all_metrics(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def to_dict(self) -> Dict[str, Union[str, float]]:
        raise NotImplementedError


class LossChecker(MetricChecker):
    def __init__(
        self,
        loss: float,
        loss_type: LossType = LossType.LOSS,
        metric_type: Optional[MetricType] = None,
    ):
        self.loss = loss
        self.loss_type = loss_type
        self.metric_type = metric_type
        self.found_loss = False

    def found_all_metrics(self) -> bool:
        return self.found_loss

    def should_check(self, log_line: str) -> bool:
        if self.loss_type == LossType.LOSS:
            return "losses_distributed" in log_line

        # all other loss types:
        assert self.metric_type is not None
        return f"Client {self.metric_type.value} Losses" in log_line

    def check(self, log_line: str) -> None:
        has_loss = False
        if self.loss_type == LossType.LOSS:
            has_loss = "losses_distributed" in log_line and str(self.loss) in log_line
        else:
            has_loss = f"{self.loss_type.value}: {self.loss}" in log_line

        if has_loss:
            self.found_loss = True

    def to_dict(self) -> Dict[str, Union[str, float]]:
        return {
            "loss": self.loss,
            "loss_type": self.loss_type.value,
            "metric_type": self.metric_type.value if self.metric_type is not None else "",
        }


class AccuracyChecker(MetricChecker):
    def __init__(
        self,
        accuracy: float,
        metric_type: MetricType,
        scope: MetricScope = MetricScope.PREDICTION,
    ):
        self.metric_type = metric_type
        self.accuracy = accuracy
        self.scope = scope
        self.found_accuracy = False

    def should_check(self, log_line: str) -> bool:
        if self.metric_type == MetricType.TRAINING:
            has_metrics = "metrics_distributed_fit" in log_line
        else:
            has_metrics = "metrics_distributed" in log_line

        has_metrics |= f"Client {self.metric_type.value} Metrics" in log_line

        return has_metrics

    def check(self, log_line: str) -> None:
        if self.metric_type == MetricType.TRAINING:
            has_accuracy = f"train - {self.scope.value} - accuracy" in log_line and str(self.accuracy) in log_line
        else:
            has_accuracy = f"val - {self.scope.value} - accuracy" in log_line and str(self.accuracy) in log_line

        if has_accuracy:
            self.found_accuracy = True

    def found_all_metrics(self) -> bool:
        return self.found_accuracy

    def to_dict(self) -> Dict[str, Union[float, str]]:
        return {
            "accuracy": self.accuracy,
            "metric_type": self.metric_type.value,
            "scope": self.scope.value,
        }
