import datetime
import json
import os
import uuid
from logging import INFO
from typing import Any, Dict, Optional

from flwr.common.logger import log


class MetricsReporter:
    """
    Stores metrics for a training execution and saves it to a JSON file.
    """

    def __init__(
        self,
        run_id: Optional[str] = None,
        output_folder: str = "metrics",
    ):
        """
        Args:
            run_id (str): the identifier for the run which these metrics are from.
                Optional, default is a random UUID.
            output_folder (str): the folder to save the metrics to. The metrics will be saved in a file
                named {output_folder}/{run_id}.json. Optional, default is "metrics".
        """
        if run_id is not None:
            self.run_id = run_id
        else:
            self.run_id = str(uuid.uuid4())

        self.output_folder = output_folder
        self.metrics: Dict[str, Any] = {}

    def add_to_metrics(self, data: Dict[str, Any]) -> None:
        """
        Adds a dictionary of data into the main metrics dictionary.

        Args:
            data (Dict[str, Any]): Data to be added to the metrics dictionary via .update().
        """
        self.metrics.update(data)

    def add_to_metrics_at_round(self, fl_round: int, data: Dict[str, Any]) -> None:
        """
        Adds a dictionary of data into the metrics dictionary for a specific FL round.

        Args:
            fl_round (int): the FL round these metrics are from.
            data (Dict[str, Any]): Data to be added to the round's metrics dictionary via .update().
        """
        if "rounds" not in self.metrics:
            self.metrics["rounds"] = {}

        if fl_round not in self.metrics["rounds"]:
            self.metrics["rounds"][fl_round] = {}

        self.metrics["rounds"][fl_round].update(data)

    def dump(self) -> None:
        """
        Dumps the current metrics to a JSON file at {self.output_folder}/{self.run_id}.json
        """
        output_file_path = os.path.join(self.output_folder, f"{self.run_id}.json")
        log(INFO, f"Dumping metrics to {output_file_path}")

        if not os.path.isdir(self.output_folder):
            os.mkdir(self.output_folder)

        with open(output_file_path, "w") as output_file:
            json.dump(self.metrics, output_file, indent=4, cls=DateTimeEncoder)


class DateTimeEncoder(json.JSONEncoder):
    """
    Converts a datetime object to string in order to make json encoding easier.
    """

    def default(self, obj: Any) -> Any:
        if isinstance(obj, datetime.datetime):
            return str(obj)
        else:
            return json.JSONEncoder.default(self, obj)
