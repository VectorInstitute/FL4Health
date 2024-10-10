import datetime
import json
import uuid
from logging import INFO
from pathlib import Path
from typing import Any

from flwr.common.logger import log

from fl4health.reporting.base_reporter import BaseReporter


class DateTimeEncoder(json.JSONEncoder):
    """
    Converts a datetime object to string in order to make json encoding easier.
    """

    def default(self, obj: Any) -> Any:
        if isinstance(obj, datetime.datetime):
            return str(obj)
        else:
            return json.JSONEncoder.default(self, obj)


class FileReporter(BaseReporter):
    def __init__(
        self,
        run_id: str | None = None,
        output_folder: str | Path = Path("metrics"),
    ):
        """Reports data each round and saves as a json.

        Args:
            run_id (str | None, optional): the identifier for the run which these
                metrics are from. If left as None will check if an id is provided during
                initialize, otherwise uses a UUID.
            output_folder (str | Path): the folder to save the metrics to. The metrics
                will be saved in a file named {output_folder}/{run_id}.json. Optional,
                default is "metrics".
        """
        self.run_id = run_id

        self.output_folder = Path(output_folder)
        self.metrics: dict[str, Any] = {}

        self.output_folder.mkdir(exist_ok=True)
        assert self.output_folder.is_dir(), f"Output folder '{self.output_folder}' is not a valid directory."

    def initialize(self, **kwargs: Any) -> None:
        # If run_id was not specified on init try first to initialize with client name
        if self.run_id is None:
            self.run_id = kwargs.get("id")
        # If client name was not provided, init run id manually
        if self.run_id is None:
            self.run_id = str(uuid.uuid4())

    def report(
        self,
        data: dict,
        round: int | None = None,
        epoch: int | None = None,
        batch: int | None = None,
    ) -> None:
        if round is None:  # Reports outside of a fit round
            self.metrics.update(data)
        # Ensure we don't report for each epoch or step
        elif epoch is None and batch is None:
            if "rounds" not in self.metrics:
                self.metrics["rounds"] = {}
            if round not in self.metrics["rounds"]:
                self.metrics["rounds"][round] = {}

            self.metrics["rounds"][round].update(data)

    def dump(self) -> None:
        raise NotImplementedError

    def shutdown(self) -> None:
        self.dump()


class JsonReporter(FileReporter):
    def dump(self) -> None:
        assert isinstance(self.run_id, str)
        """Dumps the current metrics to a JSON file at {output_folder}/{run_id.json}"""
        output_file_path = Path(self.output_folder, self.run_id).with_suffix(".json")
        log(INFO, f"Dumping metrics to {output_file_path}")

        with open(output_file_path, "w") as output_file:
            json.dump(self.metrics, output_file, indent=4)
