import tempfile
from typing import Iterator

import pytest


@pytest.fixture(scope="session")
def test_checkpoint_dirname() -> Iterator[str]:
    tempdir = tempfile.TemporaryDirectory()
    yield tempdir.name
    tempdir.cleanup()
