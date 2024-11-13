from enum import Enum
from pathlib import Path
from typing import Any

import wandb
import wandb.wandb_run

from fl4health.reporting.base_reporter import BaseReporter


class StepType(Enum):
    ROUND = "round"
    EPOCH = "epoch"
    STEP = "step"


# TODO: Add ability to parse data types and save certain data types in specific ways
# (eg. Artifacts, Tables, etc.)
class WandBReporter(BaseReporter):
    def __init__(
        self,
        wandb_step_type: StepType | str = StepType.ROUND,
        project: str | None = None,
        entity: str | None = None,
        config: dict | str | None = None,
        group: str | None = None,
        job_type: str | None = None,
        tags: list[str] | None = None,
        name: str | None = None,
        id: str | None = None,
        **kwargs: Any
    ) -> None:
        """
        _summary_

        Args:
            wandb_step_type (StepType | str, optional): How frequently to log data. Either every 'round', 'epoch' or
                'step'. Defaults to StepType.ROUND.
            project (str | None, optional): The name of the project where you're sending the new run. If unspecified,
                wandb will try to infer or set to "uncategorized"
            entity (str | None, optional): An entity is a username or team name where you're sending runs. This entity
                must exist before you can send runs there, so make sure to create your account or team in the UI before
                starting to log runs. If you don't specify an entity, the run will be sent to your default entity.
                Change your default entity in your settings under "default location to create new projects".
            config (str | None, optional): This sets wandb.config, a dictionary-like object for saving inputs to your
                job such as hyperparameters for a model. If dict: will load the key value pairs into the wandb.config
                object. If str: will look for a yaml file by that name, and load config from that file into the
                wandb.config object.
            group (str | None, optional): Specify a group to organize individual runs into a larger experiment.
            job_type (str | None, optional): Specify the type of run, useful when grouping runs.
            tags (list[str] |None, optional): A list of strings, which will populate the list of tags on this run. If
                you want to add tags to a resumed run without overwriting its existing tags, use run.tags +=
                ["new_tag"] after wandb.init().
            name (str | None, optional): A short display name for this run. Default generates a random two-word name.
            id (str | None, optional): A unique ID for this run. It must be unique in the project, and if you delete a
                run you can't reuse the ID.
            kwargs (Any):  Keyword arguments to wandb.init excluding the ones explicitly described above.
                Documentation here: https://docs.wandb.ai/ref/python/init/
        """

        # Create wandb metadata dir if necessary
        if kwargs.get("dir") is not None:
            Path(kwargs["dir"]).mkdir(exist_ok=True)

        # Set attributes
        self.wandb_init_kwargs = kwargs
        self.wandb_step_type = StepType(wandb_step_type)
        self.run_started = False
        self.initialized = False
        self.project = project
        self.entity = entity
        self.config = config
        self.group = group
        self.job_type = job_type
        self.tags = tags
        self.name = name
        self.id = id

        # Initialize run later to avoid creating runs while debugging
        self.run: wandb.wandb_run.Run

    def initialize(self, **kwargs: Any) -> None:
        """Checks if an id was provided by the client or server.

        If an id was passed to the WandBReporter on init then it takes priority over the
        one passed by the client/server.
        """
        if self.id is None:
            self.id = kwargs.get("id")
            self.initialized = True

    def start_run(self, wandb_init_kwargs: dict[str, Any]) -> None:
        """Initializes the wandb run.

        Args:
            wandb_init_kwargs (dict[str, Any]): Keyword arguments for wandb.init() excluding the ones explicitly
                accessible through WandBReporter.init().
        """
        if not self.initialized:
            self.initialize()

        self.run = wandb.init(
            project=self.project,
            entity=self.entity,
            config=self.config,
            group=self.group,
            job_type=self.job_type,
            tags=self.tags,
            name=self.name,
            id=self.id,
            **wandb_init_kwargs  # Other less commonly used kwargs
        )
        self.run_id = self.run._run_id  # If run_id was None, we need to reset run id
        self.run_started = True

    def get_wandb_timestep(
        self,
        round: int | None,
        epoch: int | None,
        step: int | None,
    ) -> int | None:
        """Determines the current step based on the timestep type.

        The report method is called every round epoch and step by default. Depending on the wandb_step_type we need to
        determine whether or not to ignore the call to avoid reporting to frequently. E.g. if wandb_step_type is EPOCH
        then we should not report data that is sent every step, but we should report data that is sent once an epoch or
        once a round. We can do this by ignoring calls to report where step is not None.

        Args:
            round (int | None): The current round or None if called outside of a round.
            epoch (int | None): The current epoch or None if called outside of a epoch.
            step (int | None): The current step (total) or None if called outside of
                step.

        Returns:
            int | None: Returns None if the reporter should not report metrics on this
                call. If an integer is returned then it is what the reporter should use
                as the current wandb step based on its wandb_step_type.
        """
        if self.wandb_step_type == StepType.ROUND and epoch is None and step is None:
            return round  # If epoch or step are integers, we should ignore report whend wandb_step_type is ROUND
        elif self.wandb_step_type == StepType.EPOCH and step is None:
            return epoch  # If step is an integer, we should ignore report when wandb_step_type is EPOCH or ROUND
        elif self.wandb_step_type == StepType.STEP:
            return step  # Since step is the finest granularity step type, we always report for wandb_step_type STEP

        # Return None otherwise
        return None

    def report(
        self,
        data: dict,
        round: int | None = None,
        epoch: int | None = None,
        batch: int | None = None,
    ) -> None:
        # If round is None, assume data is summary information. Always report this.
        if round is None:
            if not self.run_started:
                self.start_run(self.wandb_init_kwargs)
            self.run.summary.update(data)

        # Get wandb step based on timestep_type
        wandb_step = self.get_wandb_timestep(round, epoch, batch)

        # If wandb_step is None, then we should not report on this call
        if wandb_step is None:
            return

        # Check if wandb run has been initialized
        if not self.run_started:
            self.start_run(self.wandb_init_kwargs)

        # Log data
        self.run.log(data, step=wandb_step)

    def shutdown(self) -> None:
        self.run.finish()
