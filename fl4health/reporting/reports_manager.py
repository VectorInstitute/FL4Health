from collections.abc import Sequence
from typing import Any

from fl4health.reporting.base_reporter import BaseReporter


class ReportsManager:
    def __init__(self, reporters: Sequence[BaseReporter] | None = None) -> None:
        """
        Basic class for managing sequences of reporters. Generally, this class orchestrates initializing, calling,
        and gracefully shutting down all reporters provided to the class.

        Args:
            reporters (Sequence[BaseReporter] | None, optional): Reporters of ``BaseReporter`` to be managed by
                this class. Defaults to None.
        """
        self.reporters = [] if reporters is None else list(reporters)

    def initialize(self, **kwargs: Any) -> None:
        for r in self.reporters:
            r.initialize(**kwargs)

    def report(self, data: dict, round: int | None = None, epoch: int | None = None, step: int | None = None) -> None:
        for r in self.reporters:
            r.report(data, round, epoch, step)

    def shutdown(self) -> None:
        for r in self.reporters:
            r.shutdown()
