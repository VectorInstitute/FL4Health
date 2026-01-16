from enum import Enum
from logging import WARNING
from pathlib import Path
from typing import Any, Literal

import wandb
import wandb.wandb_run
from flwr.common.logger import log

from fl4health.reporting.base_reporter import BaseReporter


class WandBStepType(Enum):
    ROUND = "round"
    EPOCH = "epoch"
    STEP = "step"


# TODO: Add ability to parse data types and save certain data types in specific ways
# (e.g. Artifacts, Tables, etc.)
class WandBReporter(BaseReporter):
    def __init__(
        self,
        wandb_step_type: WandBStepType | str = WandBStepType.ROUND,
        project: str | None = None,
        entity: str | None = None,
        config: dict | str | None = None,
        group: str | None = None,
        job_type: str | None = None,
        tags: list[str] | None = None,
        name: str | None = None,
        id: str | None = None,
        resume: Literal["allow", "never", "must", "auto"] | bool | None = "allow",
        **kwargs: Any,
    ) -> None:
        """
        Weights and Biases Reporter for logging experimental results associated with FL runs.

        Args:
            wandb_step_type (StepType | str, optional): Whether to use the "round", "epoch" or "step" as the
                ``wandb_step`` value when logging information to the wandb server.
            project (str | None, optional): The name of the project where you're sending the new run. If unspecified,
                wandb will try to infer or set to "uncategorized"
            entity (str | None, optional): An entity is a username or team name where you're sending runs. This entity
                must exist before you can send runs there, so make sure to create your account or team in the UI before
                starting to log runs. If you do not specify an entity, the run will be sent to your default entity.
                Change your default entity in your settings under "default location to create new projects".
            config (str | None, optional): This sets ``wandb.config``, a dictionary-like object for saving inputs to
                your job such as hyperparameters for a model.

                - If ``dict``: will load the key value pairs into the  ``wandb.config`` object.
                - If ``str``: will look for a yaml file by that name, and load config from that file into the
                  ``wandb.config`` object.
            group (str | None, optional): Specify a group to organize individual runs into a larger experiment.
            job_type (str | None, optional): Specify the type of run, useful when grouping runs.
            tags (list[str] |None, optional): A list of strings, which will populate the list of tags on this run. If
                you want to add tags to a resumed run without overwriting its existing tags, use ``run.tags +=
                ["new_tag"]`` after ``wandb.init()``.
            name (str | None, optional): A short display name for this run. Default generates a random two-word name.
            id (str | None, optional): A unique ID for this run. It must be unique in the project, and if you delete a
                run you cannot reuse the ID.
            resume (str): Indicates how to handle the case when a run has the same entity, project and run id as
                a previous run. "must" enforces the run must resume from the run with same id and throws an error
                if it does not exist. "never" enforces that a run will not resume and throws an error if run id exists.
                "allow" resumes if the run id already exists. Defaults to "allow".
            kwargs (Any): Keyword arguments to ``wandb.init`` excluding the ones explicitly described above.
                Documentation here: https://docs.wandb.ai/ref/python/init/
        """
        # Create wandb metadata dir if necessary
        if kwargs.get("dir") is not None:
            Path(kwargs["dir"]).mkdir(exist_ok=True)

        # Set attributes
        self.wandb_init_kwargs = kwargs
        self.wandb_step_type = WandBStepType(wandb_step_type)
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
        self.resume = resume

        # Keep track of epoch and step. Initialize as 0.
        self.current_epoch = 0
        self.current_step = 0

        # Initialize run later to avoid creating runs while debugging
        self.run: wandb.wandb_run.Run

    def initialize(self, **kwargs: Any) -> None:
        """
        Checks if an id was provided by the client or server.

        If an id was passed to the ``WandBReporter`` on init then it takes priority over the one passed by the
        client/server.
        """
        if self.id is None:
            self.id = kwargs.get("id")

        if self.name is None:
            self.name = kwargs.get("name")

        self.initialized = True

    def define_metrics(self) -> None:
        """
        This method defines some of the metrics we expect to see from ``BasicClient`` and server.

        **NOTE** that you do not have to define metrics, but it can be useful for determining what should and
        shouldn't go into the run summary.
        """
        # Note that the hidden argument is not working. Raised issue here: https://github.com/wandb/wandb/issues/8890
        # Round, epoch and step
        self.run.define_metric("fit_step", summary="none", hidden=True)  # Current fit step
        self.run.define_metric("fit_epoch", summary="none", hidden=True)  # Current fit epoch
        self.run.define_metric("round", summary="none", hidden=True)  # Current server round
        self.run.define_metric("round_start", summary="none", hidden=True)
        self.run.define_metric("round_end", summary="none", hidden=True)
        # A server round contains a fit_round and maybe also an evaluate round
        self.run.define_metric("fit_round_start", summary="none", hidden=True)
        self.run.define_metric("fit_round_end", summary="none", hidden=True)
        self.run.define_metric("eval_round_start", summary="none", hidden=True)
        self.run.define_metric("eval_round_end", summary="none", hidden=True)
        # The metrics computed on all the samples from the final epoch, or the entire round if training by steps
        self.run.define_metric("fit_round_time_elapsed", summary="none")
        self.run.define_metric("eval_round_time_elapsed", summary="none")
        self.run.define_metric("fit_round_metrics", step_metric="round", summary="best")
        self.run.define_metric("eval_round_metrics", step_metric="round", summary="best")
        # Average of the losses for each step in the final epoch, or the entire round if training by steps.
        self.run.define_metric("fit_round_losses", step_metric="round", summary="best", goal="minimize")
        self.run.define_metric("eval_round_loss", step_metric="round", summary="best", goal="minimize")
        # The metrics computed  at the end of the epoch on all the samples from the epoch
        self.run.define_metric("fit_round_metrics", step_metric="fit_epoch", summary="best")
        # Average of the losses for each step in the epoch
        self.run.define_metric("fit_epoch_losses", step_metric="fit_epoch", summary="best", goal="minimize")
        # The loss and metrics for each individual step
        self.run.define_metric("fit_step_metrics", step_metric="fit_step", summary="best")
        self.run.define_metric("fit_step_losses", step_metric="fit_step", summary="best", goal="minimize")
        # FlServer (Base Server) specific metrics
        self.run.define_metric("val - loss - aggregated", step_metric="round", summary="best", goal="minimize")
        self.run.define_metric("eval_round_metrics_aggregated", step_metric="round", summary="best")
        # The following metrics don't work with wandb since they are currently obtained after training instead of live
        self.run.define_metric("val - loss - centralized", step_metric="round", summary="best", goal="minimize")
        self.run.define_metric("eval_round_metrics_centralized", step_metric="round", summary="best")

    def start_run(self, wandb_init_kwargs: dict[str, Any]) -> None:
        """
        Initializes the wandb run.

        We avoid doing this in the ``self.init`` function so that when debugging, jobs that fail before training
        starts do not get uploaded to wandb.

        Args:
            wandb_init_kwargs (dict[str, Any]): Keyword arguments for ``wandb.init()`` excluding the ones explicitly
                accessible through ``WandBReporter.init()``.
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
            resume=self.resume,
            **wandb_init_kwargs,  # Other less commonly used kwargs
        )
        self.run_id = self.run.id  # If run_id was None, we need to reset run id
        self.run_started = True

        # Wandb metric definitions
        self.define_metrics()

    def report(
        self,
        data: dict[str, Any],
        round: int | None = None,
        epoch: int | None = None,
        step: int | None = None,
    ) -> None:
        """
        Reports wandb compatible data to the wandb server.

        Data passed to ``self.report`` is always reported. If round is None, the data is reported as config
        information. If round is specified, the data is logged to the wandb run at the current wandb step which is
        either the current round, epoch or step depending on the ``wandb_step_type`` passed on initialization. The
        current epoch  and step are initialized at 0 and updated internally when specified as arguments to report.
        Therefore leaving  epoch or step as None will overwrite the data for the previous epoch/step if the key is the
        same, otherwise  the new key-value pairs are added. For example, if ``{"loss": value}`` is logged every epoch
        but  ``wandb_step_type`` is "round", then the value for "loss" at round 1 will be it's value at the last epoch
        of  that round. You can only update or overwrite the current wandb step, previous steps can not be modified.

        Args:
            data (dict[str, Any]): Dictionary of wandb compatible data to log.
            round (int | None, optional): The current FL round. If None, this indicates that the method was called
                outside of a round (e.g. for summary information). Defaults to None.
            epoch (int | None, optional): The current epoch (In total across all rounds). If None then this method was
                not called at or within the scope of an epoch. Defaults to None.
            step (int | None, optional): The current step (In total across all rounds and epochs). If None then this
                method was called outside the scope of a training or evaluation step (e.g. at the end of an epoch or
                round) Defaults to None.
        """
        # Now that report has been called we are finally forced to start the run.
        if not self.run_started:
            self.start_run(self.wandb_init_kwargs)

        # If round is None, assume data is summary information.
        if round is None:
            wandb.config.update(data)
            return

        # Update current epoch and step if they were specified
        if epoch is not None:
            if epoch < self.current_epoch:
                log(
                    WARNING,
                    f"The specified current epoch ({epoch}) is less than a previous \
                        current epoch ({self.current_epoch})",
                )
            self.current_epoch = epoch

        if step is not None:
            if step < self.current_step:
                log(
                    WARNING,
                    f"The specified current step ({step}) is less than a previous current step ({self.current_step})",
                )
            self.current_step = step

        # Log based on step type
        if self.wandb_step_type == WandBStepType.ROUND:
            self.run.log(data, step=round)
        elif self.wandb_step_type == WandBStepType.EPOCH:
            self.run.log(data, step=self.current_epoch)
        elif self.wandb_step_type == WandBStepType.STEP:
            self.run.log(data, step=self.current_step)

    def shutdown(self) -> None:
        self.run.finish()
