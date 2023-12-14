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
        loss: Optional[float] = None,
        checkpoint_loss: Optional[float] = None,
        backward_loss: Optional[float] = None,
        proximal_loss: Optional[float] = None,
        global_loss: Optional[float] = None,
        local_loss: Optional[float] = None,
        metric_type: Optional[MetricType] = None,
    ):
        self.loss = loss
        self.checkpoint_loss = checkpoint_loss
        self.backward_loss = backward_loss
        self.proximal_loss = proximal_loss
        self.global_loss = global_loss
        self.local_loss = local_loss
        self.metric_type = metric_type
        self.found_losses = False

    def found_all_metrics(self) -> bool:
        return self.found_losses

    def should_check(self, log_line: str) -> bool:
        has_any_loss = False
        if self.loss is not None:
            has_any_loss |= "losses_distributed" in log_line
        if (
            self.checkpoint_loss is not None
            or self.backward_loss is not None
            or self.global_loss is not None
            or self.local_loss is not None
        ):
            assert self.metric_type is not None
            has_any_loss |= f"Client {self.metric_type.value} Losses" in log_line
        return has_any_loss

    def check(self, log_line: str) -> None:
        if self.loss is not None:
            has_loss = "losses_distributed" in log_line and str(self.loss) in log_line
        else:
            has_loss = True

        if self.checkpoint_loss is not None:
            has_checkpoint_loss = f"checkpoint: {self.checkpoint_loss}" in log_line
        else:
            has_checkpoint_loss = True

        if self.backward_loss is not None:
            has_backward_loss = f"backward: {self.backward_loss}" in log_line
        else:
            has_backward_loss = True

        if self.proximal_loss is not None:
            has_proximal_loss = f"proximal_loss: {self.proximal_loss}" in log_line
        else:
            has_proximal_loss = True

        if self.global_loss is not None:
            has_global_loss = f"global: {self.global_loss}" in log_line
        else:
            has_global_loss = True

        if self.local_loss is not None:
            has_local_loss = f"local: {self.local_loss}" in log_line
        else:
            has_local_loss = True

        if all([has_loss, has_checkpoint_loss, has_backward_loss, has_proximal_loss, has_global_loss, has_local_loss]):
            self.found_losses = True

    def to_dict(self) -> Dict[str, Union[str, float]]:
        return_dict: Dict[str, Union[str, float]] = {}

        if self.loss is not None:
            return_dict["loss"] = self.loss
        if self.checkpoint_loss is not None:
            return_dict["checkpoint_loss"] = self.checkpoint_loss
        if self.backward_loss is not None:
            return_dict["backward_loss"] = self.backward_loss
        if self.proximal_loss is not None:
            return_dict["proximal_loss"] = self.proximal_loss
        if self.global_loss is not None:
            return_dict["global_loss"] = self.global_loss
        if self.local_loss is not None:
            return_dict["local_loss"] = self.local_loss
        if self.metric_type is not None:
            return_dict["metric_type"] = self.metric_type.value

        return return_dict


class AccuracyChecker(MetricChecker):
    def __init__(
        self,
        metric_type: MetricType,
        accuracy: float,
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
