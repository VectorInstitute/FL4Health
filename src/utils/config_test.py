from pathlib import Path

from pytest import raises

from src.utils.config import InvalidConfigError, check_config, load_config


def test_load_config() -> None:
    """Ensure loaded config is non empty"""
    config_path = f"{Path(__file__).parent.parent}/examples/basic_example/config.yaml"
    config = load_config(config_path)
    assert len(config.keys()) != 0


def test_check_config() -> None:
    """Verify invalid configs raise Error"""

    # Test missing values
    config = {"n_clients": 5, "n_server_rounds": 10}

    with raises(InvalidConfigError):
        check_config(config)

    # Test incorrect value type
    config = {"n_clients": 5, "n_server_rounds": 10, "local_epochs": 5, "batch_size": 45.8}  # type: ignore

    with raises(InvalidConfigError):
        check_config(config)

    # Test invalid value range
    config = {"n_clients": 5, "n_server_rounds": 10, "local_epochs": -1, "batch_size": 4}

    with raises(InvalidConfigError):
        check_config(config)
