from pathlib import Path

from flwr.common.typing import Config
from pytest import raises

from fl4health.utils.config import InvalidConfigError, check_config, load_config, make_dict_with_epochs_or_steps


def test_load_config() -> None:
    """Ensure loaded config is non empty."""
    config_path = f"{Path(__file__).parent}/resources/config.yaml"
    config = load_config(config_path)
    assert len(config.keys()) != 0


def test_check_config() -> None:
    """Verify invalid configs raise Error."""
    # Test missing values
    config_1: Config = {"n_server_rounds": 10}

    with raises(InvalidConfigError):
        check_config(config_1)

    # Test incorrect value type
    config_2: Config = {"n_server_rounds": 10, "batch_size": 45.8}

    with raises(InvalidConfigError):
        check_config(config_2)

    # Test invalid parameter range
    config_3: Config = {"n_server_rounds": 4.5, "batch_size": 4}

    with raises(InvalidConfigError):
        check_config(config_3)

    # Test invalid parameter range
    config_4: Config = {"n_server_rounds": 4, "batch_size": -4}

    with raises(InvalidConfigError):
        check_config(config_4)


def test_make_dict_with_epochs_or_steps() -> None:
    # Should be an empty dict because both are none
    d = make_dict_with_epochs_or_steps(local_epochs=None, local_steps=None)
    assert len(d) == 0

    # Should be a dict of length one with key local_epochs
    d = make_dict_with_epochs_or_steps(local_epochs=5, local_steps=None)
    assert len(d) == 1
    assert d["local_epochs"] == 5

    # Should be a dict of length one with key local_steps
    d = make_dict_with_epochs_or_steps(local_epochs=None, local_steps=10)
    assert len(d) == 1
    assert d["local_steps"] == 10

    # Should be a dict of length one with key local_epochs since that takes precedence
    d = make_dict_with_epochs_or_steps(local_epochs=5, local_steps=10)
    assert len(d) == 1
    assert d["local_epochs"] == 5
