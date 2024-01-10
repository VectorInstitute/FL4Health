import datetime
import json
import os
from logging import DEBUG, ERROR, INFO
from typing import Any, Dict, Optional

from flwr.common.logger import log


class MetricsReporter:
    def __init__(
        self,
        run_id: str = "default",
        output_folder: str = "metrics",
        num_rounds: Optional[int] = None,
        dump_at_every_step: bool = False,
    ):
        # TODO docstrings
        self.run_id = run_id
        self.output_folder = output_folder
        self.num_rounds = num_rounds
        self.dump_at_every_step = dump_at_every_step
        self.metrics: Dict[str, Any] = {}

    def add_to_metrics(self, data: Dict[str, Any]) -> None:
        # TODO docstrings
        self.metrics.update(data)

        if self.dump_at_every_step:
            self.dump()

    def add_to_metrics_at_round(self, round: int, data: Dict[str, Any]) -> None:
        # TODO docstrings
        if self.num_rounds is None:
            log(DEBUG, "MetricsReporter: num_rounds is None, adding metrics to a new round every time")
            if "rounds" not in self.metrics:
                self.metrics["rounds"] = []

            self.metrics["rounds"].add(data)
        else:
            if "rounds" not in self.metrics:
                self.metrics["rounds"] = [{}] * self.num_rounds

            if round - 1 < 0:
                log(ERROR, f"MetricsReporter: round value is invalid ({round})")
                return

            self.metrics["rounds"][round - 1].update(data)

        if self.dump_at_every_step:
            self.dump()

    def dump(self) -> None:
        output_file_path = os.path.join(self.output_folder, f"{self.run_id}.json")
        log(INFO, f"Dumping metrics to {output_file_path}")

        if not os.path.isdir(self.output_folder):
            os.mkdir(self.output_folder)

        with open(output_file_path, "w") as output_file:
            json.dump(self.metrics, output_file, indent=4, cls=DateTimeEncoder)


class DateTimeEncoder(json.JSONEncoder):
    # TODO docstrings
    def default(self, obj: Any) -> Any:
        if isinstance(obj, datetime.datetime):
            return str(obj)
        else:
            return json.JSONEncoder.default(self, obj)
