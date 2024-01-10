import datetime
import json
import os
import uuid
from logging import INFO
from typing import Any, Dict, Optional

from flwr.common.logger import log


class MetricsReporter:
    def __init__(
        self,
        run_id: Optional[str] = None,
        output_folder: str = "metrics",
        dump_at_every_step: bool = False,
    ):
        # TODO docstrings
        if run_id is not None:
            self.run_id = run_id
        else:
            self.run_id = str(uuid.uuid4())

        self.output_folder = output_folder
        self.dump_at_every_step = dump_at_every_step
        self.metrics: Dict[str, Any] = {}

    def add_to_metrics(self, data: Dict[str, Any]) -> None:
        # TODO docstrings
        self.metrics.update(data)

        if self.dump_at_every_step:
            self.dump()

    def add_to_metrics_at_round(self, round: int, data: Dict[str, Any]) -> None:
        # TODO docstrings
        if "rounds" not in self.metrics:
            self.metrics["rounds"] = {}

        if round not in self.metrics["rounds"]:
            self.metrics["rounds"][round] = {}

        self.metrics["rounds"][round].update(data)

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
