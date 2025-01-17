import tempfile
from typing import Iterator

import pytest
import torch


@pytest.fixture(scope="session")
def test_checkpoint_dirname() -> Iterator[str]:
    tempdir = tempfile.TemporaryDirectory()
    yield tempdir.name
    tempdir.cleanup()


@pytest.fixture(scope="session")
def tolerance() -> float:
    if torch.cuda.is_available():
        return 0.005
    else:
        return 0.0005
