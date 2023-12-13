from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Optional


class MetricType(Enum):
    TRAINING = "Training"
    VALIDATION = "Validation"


class MetricsChecker(ABC):
    def __init__(self, metric_type: MetricType, accuracy: float, loss: float):
        self.metric_type = metric_type
        self.accuracy = accuracy
        self.loss = loss

    def assert_metrics_exist(self, logs: str) -> None:
        log_lines = logs.split("\n")
        for log_line in log_lines:
            if self.should_assert(log_line):
                self.check_metrics(log_line)

        assert self.found_all_metrics(), (
            f"Full output:\n{logs}\n" f"[ASSERT ERROR] Metrics {self.to_dict()} not found in logs."
        )

    @abstractmethod
    def should_assert(self, log_line: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def check_metrics(self, log_line: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def found_all_metrics(self) -> bool:
        raise NotImplementedError

    def to_dict(self) -> Dict[str, str]:
        return {
            "metric_type": self.metric_type.value,
            "loss": str(self.loss),
            "accuracy": str(self.accuracy),
        }


class ServerMetricsChecker(MetricsChecker):
    def __init__(self, metric_type: MetricType, accuracy: float, loss: Optional[float] = None):
        super().__init__(metric_type, accuracy, loss)  # type: ignore
        self.accuracy = accuracy
        self.loss = loss  # type: ignore
        self.found_loss = False
        self.found_accuracy = False

    def should_assert(self, log_line: str) -> bool:
        has_losses = "losses_distributed" in log_line
        if self.metric_type == MetricType.TRAINING:
            has_metrics = "metrics_distributed_fit" in log_line
        else:
            has_metrics = "metrics_distributed" in log_line
        return has_losses or has_metrics

    def check_metrics(self, log_line: str) -> None:
        if self.loss is not None:
            has_loss = "losses_distributed" in log_line and str(self.loss) in log_line
        else:
            has_loss = True  # if loss is not provided, assume it exists

        if self.metric_type == MetricType.TRAINING:
            has_accuracy = "train - prediction - accuracy" in log_line and str(self.accuracy) in log_line
        else:
            has_accuracy = "val - prediction - accuracy" in log_line and str(self.accuracy) in log_line

        if has_loss:
            self.found_loss = True
        if has_accuracy:
            self.found_accuracy = True

    def found_all_metrics(self) -> bool:
        return self.found_loss and self.found_accuracy


class ClientMetricsChecker(MetricsChecker):
    def __init__(self, metric_type: MetricType, accuracy: float, loss: float, backward: float):
        super().__init__(metric_type, accuracy, loss)
        self.backward = backward
        self.found_losses = False
        self.found_accuracy = False

    def should_assert(self, log_line: str) -> bool:
        has_losses = f"Client {self.metric_type.value} Losses" in log_line
        has_metrics = f"Client {self.metric_type.value} Metrics" in log_line
        return has_losses or has_metrics

    def check_metrics(self, log_line: str) -> None:
        has_checkpoint = f"checkpoint: {self.loss}" in log_line
        has_backward = f"backward: {self.backward}" in log_line
        has_accuracy = f"accuracy: {self.accuracy}" in log_line

        if has_checkpoint and has_backward:
            self.found_losses = True
        if has_accuracy:
            self.found_accuracy = True

    def found_all_metrics(self) -> bool:
        return self.found_losses and self.found_accuracy

    def to_dict(self) -> Dict[str, str]:
        dictionary = super().to_dict()
        return {
            **dictionary,
            "backward": str(self.backward),
        }


class ProxClientMetricsChecker(ClientMetricsChecker):
    def __init__(self, metric_type: MetricType, accuracy: float, loss: float, backward: float, proximal_loss: float):
        super().__init__(metric_type, accuracy, loss, backward)
        self.proximal_loss = proximal_loss
        self.found_proximal_loss = False

    def check_metrics(self, log_line: str) -> None:
        if not self.found_losses:
            super().check_metrics(log_line)
            has_proximal_loss = f"proximal_loss: {self.proximal_loss}" in log_line

            if self.found_losses and has_proximal_loss:
                self.found_proximal_loss = True
        else:
            super().check_metrics(log_line)

    def found_all_metrics(self) -> bool:
        found_client_metrics = super().found_all_metrics()
        return found_client_metrics and self.found_proximal_loss

    def to_dict(self) -> Dict[str, str]:
        dictionary = super().to_dict()
        return {
            **dictionary,
            "proximal_loss": str(self.proximal_loss),
        }


class LogChecker(ABC):
    @abstractmethod
    def assert_on_logs(self, logs: str) -> None:
        raise NotImplementedError
