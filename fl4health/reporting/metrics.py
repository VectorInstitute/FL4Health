from logging import DEBUG, INFO
from typing import Any, Dict, Optional

from flwr.common.logger import log


class MetricsReporter:
    def __init__(self, num_rounds: Optional[int] = None):
        # TODO docstrings
        self.num_rounds = num_rounds
        self.metrics: Dict[str, Any] = {}

    def add_to_metrics(self, **kwargs: Any) -> None:
        # TODO docstrings
        log(INFO, f"add_to_metrics: {kwargs}")
        data = _unpack(**kwargs)
        self.metrics.update(data)

    def add_to_metrics_at_round(self, round: int, **kwargs: Any) -> None:
        # TODO docstrings
        log(INFO, f"add_to_metrics_at_round: {round}, {kwargs}")
        data = _unpack(**kwargs)

        if self.num_rounds is None:
            log(DEBUG, "MetricsReporter: num_rounds is None, adding metrics to a new round every time")

            if "rounds" not in self.metrics:
                self.metrics["rounds"] = []

            self.metrics["rounds"].add(data)
        else:
            if "rounds" not in self.metrics:
                self.metrics["rounds"] = [{}] * self.num_rounds
            self.metrics["rounds"][round - 1].update(data)


def _unpack(**kwargs: Any) -> Dict[str, Any]:
    # TODO docstrings
    flattened = {}
    for key in kwargs:
        if type(kwargs[key]) is dict:
            for sub_key in kwargs[key]:
                flattened[sub_key] = kwargs[key][sub_key]
        else:
            flattened[key] = kwargs[key]

    return flattened
