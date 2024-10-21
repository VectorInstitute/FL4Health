import json
import uuid
from logging import INFO
from pathlib import Path
from typing import Any

from flwr.common.logger import log

from fl4health.reporting.base_reporter import BaseReporter


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
        self.initialized = False

        self.output_folder.mkdir(exist_ok=True)

    def initialize(self, **kwargs: Any) -> None:
        # If run_id was not specified on init try first to initialize with client name
        if self.run_id is None:
            self.run_id = kwargs.get("id")
        # If client name was not provided, init run id manually
        if self.run_id is None:
            self.run_id = str(uuid.uuid4())

        self.initialized = True

    def report(
        self,
        data: dict[str, Any],
        round: int | None = None,
        epoch: int | None = None,
        step: int | None = None,
    ) -> None:
        """A method called by clients or servers to send data to the reporter.

        The report method is called by the client/server at frequent intervals (ie step, epoch, round) and sometimes
        outside of a FL round (for high level summary data). The json reporter is hardcoded to report at the 'round'
        level and therefore ignores calls to the report method made every epoch or every step.

        Args:
            data (dict): The data to maybe report from the server or client.
            round (int | None, optional): The current FL round. If None, this indicates that the method was called
                outside of a round (e.g. for summary information). Defaults to None.
            epoch (int | None, optional): The current epoch. If None then this method was not called within the scope
                of an epoch. Defaults to None.
            step (int | None, optional): The current step (total). If None then this method was called outside the
                scope of a training or evaluation step (eg. at the end of an epoch or round) Defaults to None.
        """
        if not self.initialized:
            self.initialize()

        if round is None:  # Reports outside of a fit round
            self.metrics.update(data)
        # Ensure we don't report for each epoch or step
        elif epoch is None and step is None:
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
        """Dumps the current metrics to a JSON file at {output_folder}/{run_id.json}"""
        assert self.run_id is not None
        output_file_path = Path(self.output_folder, self.run_id).with_suffix(".json")
        log(INFO, f"Dumping metrics to {str(output_file_path)}")

        with open(output_file_path, "w") as output_file:
            json.dump(self.metrics, output_file, indent=4)
