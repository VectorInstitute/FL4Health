from typing import Any, Dict, Type, TypeVar

import yaml
from flwr.common.typing import Config

REQUIRED_CONFIG = {
    "n_server_rounds": int,
    "batch_size": int,
}

T = TypeVar("T")


class InvalidConfigError(ValueError):
    pass


def load_config(config_path: str) -> Dict[str, Any]:
    """Load Configuration Dictionary"""

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    check_config(config)

    return config


def check_config(config: Dict[str, Any]) -> None:
    """Check if Configuration Dictionary is valid"""

    # Check for presence of required keys
    for req_key in REQUIRED_CONFIG.keys():
        if req_key not in config.keys():
            raise InvalidConfigError(f"{req_key} must be specified in Config File")

    # Check for invalid parameter value types
    for req_key, val in REQUIRED_CONFIG.items():
        if not isinstance(config[req_key], val):
            raise InvalidConfigError(f"{req_key} must be of type {str(val)}")

    # Check for invalid integer parameter values
    for key in ["n_server_rounds", "batch_size"]:
        if config[key] <= 0:
            raise InvalidConfigError(f"{key} must be greater than 0")


def narrow_config_type(config: Config, config_key: str, narrow_type_to: Type[T]) -> T:
    """
    Checks if a config_key exists in config and if so, verify it is of type narrow_type_to.

    Args:
        config (Config): The config object from the server.
        config_key (str): The key to check config for.
        narrow_type_to (Type[T]): The expected type of config[config_key]

    Returns:
        T: The type-checked value at config[config_key]

    Raises:
        ValueError: If config[config_key] is not of type narrow_type_to or
            if the config_key is not present in config.
    """
    if config_key not in config:
        raise ValueError(f"{config_key} is not present in the Config.")

    config_value = config[config_key]
    if isinstance(config_value, narrow_type_to):
        return config_value
    else:
        raise ValueError(f"Provided configuration key ({config_key}) value does not have correct type")
