import os
from pathlib import Path

from fl4health.reporting import WandBReporter, WandBStepType


def test_wandb_reporter_init(tmp_path: Path) -> None:
    # Create with various step types
    wandb_reporter = WandBReporter("round")
    wandb_reporter = WandBReporter("step")
    wandb_reporter = WandBReporter("epoch")
    wandb_reporter = WandBReporter(WandBStepType.STEP)
    wandb_reporter = WandBReporter(WandBStepType.EPOCH)

    # Create with a bunch of args and kwargs
    wandb_reporter = WandBReporter(
        WandBStepType.ROUND,
        project="project",
        entity="entity",
        config={"hp1": 2.0, "hp2": 1.2},
        group="group",
        job_type="job",
        tags=["tag1", "tag2"],
        id="id1",
        name="name1",
        resume="allow",
        dir=tmp_path.joinpath("wandb_dir"),
    )

    assert os.path.isdir(tmp_path.joinpath("wandb_dir"))

    assert not wandb_reporter.initialized


def test_wandb_reporter_initialize() -> None:
    wandb_reporter_with_name_id = WandBReporter(WandBStepType.ROUND, id="id1", name="name1")
    wandb_reporter_with_name = WandBReporter(WandBStepType.ROUND, name="name1")
    wandb_reporter_with_id = WandBReporter(WandBStepType.ROUND, id="id1")
    wandb_reporter_with_none = WandBReporter(WandBStepType.ROUND)

    assert not wandb_reporter_with_name_id.initialized
    assert not wandb_reporter_with_name.initialized
    assert not wandb_reporter_with_id.initialized
    assert not wandb_reporter_with_none.initialized
    # Should not replace the id and name, as it was already specified on construction
    wandb_reporter_with_name_id.initialize(id="id2", name="name2")

    assert wandb_reporter_with_name_id.id == "id1"
    assert wandb_reporter_with_name_id.name == "name1"
    assert wandb_reporter_with_name_id.initialized

    # Should set the id but not name, as name was specified on construction
    wandb_reporter_with_name.initialize(id="id2", name="name2")

    assert wandb_reporter_with_name.id == "id2"
    assert wandb_reporter_with_name.name == "name1"
    assert wandb_reporter_with_name.initialized

    # Should set the name but not id, as id was specified on construction
    wandb_reporter_with_id.initialize(id="id2", name="name2")

    assert wandb_reporter_with_id.id == "id1"
    assert wandb_reporter_with_id.name == "name2"
    assert wandb_reporter_with_id.initialized

    # Should set id and name, as neither was specified at the start.
    wandb_reporter_with_none.initialize(id="id2", name="name2")

    assert wandb_reporter_with_none.id == "id2"
    assert wandb_reporter_with_none.name == "name2"
    assert wandb_reporter_with_none.initialized
