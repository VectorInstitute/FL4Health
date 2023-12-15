from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Optional, Union

from pytest import approx

TOLERANCE = 0.0005


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
        losses = self.parse_losses(log_line)
        has_loss = any([approx(loss, abs=TOLERANCE) == self.loss for loss in losses])

        if has_loss:
            self.found_loss = True

    def parse_losses(self, log_line: str) -> List[float]:
        try:
            if self.loss_type == LossType.LOSS:
                keyword = "losses_distributed"
                if keyword in log_line:
                    losses_string = log_line.split(keyword)[-1]
                    losses_list = eval(losses_string)
                    return [loss_tuple[1] for loss_tuple in losses_list]
            else:
                keyword = f"{self.loss_type.value}:"
                if keyword in log_line:
                    after_keyword = log_line.split(keyword)[-1]
                    loss_string = after_keyword[: after_keyword.find("\t")]
                    return [eval(loss_string)]
        except Exception:
            # if any parsing error occurs, return empty
            return []
        return []

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
        accuracies = self.parse_accuracies(log_line)
        has_accuracy = any([approx(accuracy, abs=TOLERANCE) == self.accuracy for accuracy in accuracies])

        if has_accuracy:
            self.found_accuracy = True

    def parse_accuracies(self, log_line: str) -> List[float]:
        try:
            multiple_accuracies = False
            if self.metric_type == MetricType.TRAINING:
                if "metrics_distributed_fit" in log_line:
                    keyword = "metrics_distributed_fit"
                    multiple_accuracies = True
                else:
                    keyword = f"Client {self.metric_type.value} Metrics"
            else:
                if "metrics_distributed" in log_line:
                    keyword = "metrics_distributed"
                    multiple_accuracies = True
                else:
                    keyword = f"Client {self.metric_type.value} Metrics"

            if keyword not in log_line:
                return []

            if self.metric_type == MetricType.TRAINING:
                accuracy_type_keyword = f"train - {self.scope.value} - accuracy"
            else:
                accuracy_type_keyword = f"val - {self.scope.value} - accuracy"

            if multiple_accuracies:
                accuracies_string = log_line.split(keyword)[-1]
                accuracies_dict = eval(accuracies_string)

                if accuracy_type_keyword not in accuracies_dict:
                    return []

                return [accuracies_tuple[1] for accuracies_tuple in accuracies_dict[accuracy_type_keyword]]
            else:
                if accuracy_type_keyword in log_line:
                    after_keyword = log_line.split(accuracy_type_keyword + ":")[-1]
                    tab_index = after_keyword.find("\t")
                    if tab_index != -1:
                        accuracy_string = after_keyword[:tab_index]
                    else:
                        accuracy_string = after_keyword
                    return [eval(accuracy_string)]

        except Exception:
            # if any parsing error occurs, return empty
            return []
        return []

    def found_all_metrics(self) -> bool:
        return self.found_accuracy

    def to_dict(self) -> Dict[str, Union[float, str]]:
        return {
            "accuracy": self.accuracy,
            "metric_type": self.metric_type.value,
            "scope": self.scope.value,
        }
