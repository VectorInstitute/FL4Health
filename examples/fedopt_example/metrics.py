import json
from typing import Dict, List

import numpy as np
from flwr.common.typing import Metrics
from sklearn.metrics import confusion_matrix

from examples.fedopt_example.client_data import LabelEncoder


class Outcome:
    def __init__(self, class_name: str) -> None:
        self.true_positive: int = 0
        self.false_positive: int = 0
        self.false_negative: int = 0
        self.class_name = class_name

    def get_precision(self) -> float:
        if self.true_positive == 0.0:
            return 0.0
        return self.true_positive / (self.true_positive + self.false_positive)

    def get_recall(self) -> float:
        if self.true_positive + self.false_negative == 0.0:
            return 0.0
        return self.true_positive / (self.true_positive + self.false_negative)

    def get_f1(self) -> float:
        precision = self.get_precision()
        recall = self.get_recall()
        if self.true_positive == 0.0:
            return 0.0
        return (2 * precision * recall) / (precision + recall)

    def summarize(self) -> Dict[str, float]:
        return {
            f"{self.class_name}_precision": self.get_precision(),
            f"{self.class_name}_recall": self.get_recall(),
            f"{self.class_name}_f1": self.get_f1(),
        }

    @staticmethod
    def from_results_dict(class_name: str, stats_string: str) -> "Outcome":
        outcome = Outcome(class_name)
        stats = json.loads(stats_string)
        outcome.true_positive = stats[0]
        outcome.false_positive = stats[1]
        outcome.false_negative = stats[2]
        return outcome

    @staticmethod
    def merge_outcomes(outcome_1: "Outcome", outcome_2: "Outcome") -> "Outcome":
        assert outcome_1.class_name == outcome_2.class_name
        outcome_1.true_positive += outcome_2.true_positive
        outcome_1.false_negative += outcome_2.false_negative
        outcome_1.false_positive += outcome_2.false_positive
        return outcome_1


class ServerMetrics:
    def __init__(self, true_preds: int, total_preds: int, outcomes: List[Outcome]) -> None:
        self.true_preds = true_preds
        self.total_preds = total_preds
        self.outcomes = outcomes

    def compute_metrics(self) -> Metrics:
        metrics: Metrics = {"total_accuracy": self.true_preds / self.total_preds}
        for outcome in self.outcomes:
            metrics[f"{outcome.class_name}_precision"] = outcome.get_precision()
            metrics[f"{outcome.class_name}_recall"] = outcome.get_recall()
            metrics[f"{outcome.class_name}_f1"] = outcome.get_f1()
        return metrics


class ClientMetrics:
    def __init__(self, label_encoder: LabelEncoder) -> None:
        self.true_preds = 0
        self.total_preds = 0
        self.classes = label_encoder.classes
        self.outcome_dict = self._initialize_outcomes(self.classes)
        self.label_to_class = label_encoder.label_to_class
        self.results: Metrics = {}
        self.n_classes = len(self.classes)

    def _initialize_outcomes(self, classes: List[str]) -> Dict[str, Outcome]:
        return {topic: Outcome(topic) for topic in classes}

    def summarize(self) -> str:
        sum_f1 = 0.0
        n_topics = 0
        log_string = ""
        for _, outcome in self.outcome_dict.items():
            summary_dict = outcome.summarize()
            n_topics += 1
            sum_f1 += outcome.get_f1()

            self.results[
                outcome.class_name
            ] = f"[{outcome.true_positive}, {outcome.false_positive}, {outcome.false_negative}]"

            for metric_name, metric_value in summary_dict.items():
                log_string = f"{log_string}\n{metric_name}:{str(metric_value)}"

        self.results["total_preds"] = self.total_preds
        self.results["true_preds"] = self.true_preds
        log_string = f"{log_string}\ntotal_accuracy:{str(self.true_preds/self.total_preds)}"
        log_string = f"{log_string}\naverage_f1:{str(sum_f1/n_topics)}"
        return log_string

    def update_performance(self, predictions: np.ndarray, labels: np.ndarray) -> None:
        confusion = confusion_matrix(labels, predictions, labels=range(self.n_classes))
        for i in range(self.n_classes):
            true_class = self.label_to_class[i]
            for j in range(self.n_classes):
                pred_class = self.label_to_class[j]
                # int cast is because FL metrics don't play nice with numpy int.64 types
                count = int(confusion[i][j])
                if i == j:
                    self.outcome_dict[true_class].true_positive += count
                    self.true_preds += count
                    self.total_preds += count
                else:
                    self.outcome_dict[true_class].false_negative += count
                    self.outcome_dict[pred_class].false_positive += count
                    self.total_preds += count
