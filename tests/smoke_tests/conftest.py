import pytest
import torch


@pytest.fixture(scope="session")
def tolerance() -> float:
    if torch.cuda.is_available():
        return 0.005
    else:
        return 0.0005
