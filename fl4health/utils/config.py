from typing import Any, Dict

import yaml

REQUIRED_CONFIG = {
    "n_server_rounds": int,
    "n_clients": int,
    "local_epochs": int,
    "batch_size": int,
}


class InvalidConfigError(ValueError):
    pass


def load_config(config_path: str) -> Dict[str, Any]:
    """Load Configuration Dictionairy"""

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    check_config(config)

    return config


def check_config(config: Dict[str, Any]) -> None:
    """Check if Configuration Dictionairy is valid"""

    # Check for presence of required keys
    for req_key in REQUIRED_CONFIG.keys():
        if req_key not in config.keys():
            raise InvalidConfigError(f"{req_key} must be specified in Config File")

    # Check for invalid parameter value types
    for req_key, val in REQUIRED_CONFIG.items():
        if not isinstance(config[req_key], val):
            raise InvalidConfigError(f"{req_key} must be of type {str(val)}")

    # Check for invalid integer parameter values
    for key in ["n_server_rounds", "n_clients", "local_epochs"]:
        if config[key] <= 0:
            raise InvalidConfigError(f"{key} must be greater than 0")
