from enum import Enum
from pathlib import Path
from typing import Any

import wandb
import wandb.wandb_run

from fl4health.reporting.base_reporter import BaseReporter


class StepType(Enum):
    ROUND = "round"
    EPOCH = "epoch"
    BATCH = "batch"


# TODO: Add ability to parse data types and save certain data types in specific ways
# (eg. Artifacts, Tables, etc.)


class WandBReporter(BaseReporter):
    def __init__(self, timestep: StepType | str = StepType.ROUND, **kwargs: Any) -> None:
        """Reporter that logs data to a wandb server.

        Args:
            timestep (StepType | str, optional): How frequently to log data. Either
                every 'round', 'epoch' or 'step'. Defaults to StepType.ROUND.
            **kwargs (Any):
                Keyword arguments to wandb.init

        """
        # Create wandb metadata dir if necessary
        if kwargs.get("dir") is not None:
            Path(kwargs["dir"]).mkdir(exist_ok=True)

        # Create run and set attrbutes
        self.wandb_init_kwargs = kwargs
        self.timestep_type = StepType(timestep) if isinstance(timestep, str) else timestep
        self.run_started = False
        self.initialized = False
        # To maybe be initialized later
        self.run_id = kwargs.get("id")
        self.run: wandb.wandb_run.Run

    def initialize(self, **kwargs: Any) -> None:
        """Checks if an id was provided by the client or server.

        If an id was passed to the WandBReporter on init then it takes priority over the
        one passed by the client/server.
        """
        if self.run_id is None:
            self.run_id = kwargs.get("id")
            self.initialized = True

    def start_run(self, **kwargs: Any) -> None:
        """Initializes the wandb run.

        Args:
            kwargs (Any): Keyword arguments for wandb.init()
        """
        if not self.initialized:
            self.initialize()

        self.run = wandb.init(id=self.run_id, **kwargs)
        self.run_id = self.run._run_id  # If run_id was None, we need to reset run id
        self.run_started = True

    def get_wandb_timestep(
        self,
        round: int | None,
        epoch: int | None,
        step: int | None,
    ) -> int | None:
        """Determines the current step based on the timestep type.

        Args:
            round (int | None): The current round or None if called outside of a round.
            epoch (int | None): The current epoch or None if called outside of a epoch.
            step (int | None): The current step (total) or None if called outside of
                step.

        Returns:
            int | None: Returns None if the reporter should not report metrics on this
                call. If an integer is returned then it is what the reporter should use
                as the current wandb step.
        """
        if self.timestep_type == StepType.ROUND and epoch is None and step is None:
            return round
        elif self.timestep_type == StepType.EPOCH and step is None:
            return epoch
        elif self.timestep_type == StepType.BATCH:
            return step
        return None

    def report(
        self,
        data: dict,
        round: int | None = None,
        epoch: int | None = None,
        batch: int | None = None,
    ) -> None:
        # If round is None, assume data is summary information
        if round is None:
            if not self.run_started:
                self.start_run(**self.wandb_init_kwargs)
            self.run.summary.update(data)

        # Get wandb step based on timestep_type
        timestep = self.get_wandb_timestep(round, epoch, batch)

        # If timestep is None, then we should not report on this call
        if timestep is None:
            return

        # Check if wandb run has been initialized
        if not self.run_started:
            self.start_run(**self.wandb_init_kwargs)

        # Log data
        self.run.log(data, step=timestep)

    def shutdown(self) -> None:
        self.run.finish()
