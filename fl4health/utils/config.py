from collections.abc import Callable
from typing import Any, TypeVar

import yaml

REQUIRED_CONFIG = {
    "n_server_rounds": int,
    "batch_size": int,
}

T = TypeVar("T")


class InvalidConfigError(ValueError):
    pass


def load_config(config_path: str) -> dict[str, Any]:
    """Load Configuration Dictionary"""

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    check_config(config)

    return config


def check_config(config: dict[str, Any]) -> None:
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


def narrow_dict_type(dictionary: dict[str, Any], key: str, narrow_type_to: type[T]) -> T:
    """
    Checks if a key exists in dictionary and if so, verify it is of type ``narrow_type_to``.

    Args:
        dictionary (dict[str, Any]): A dictionary with string keys.
        key (str): The key to check dictionary for.
        narrow_type_to (type[T]): The expected type of dictionary[key]

    Returns:
        T: The type-checked value at dictionary[key]

    Raises:
        ValueError: If dictionary[key] is not of type ``narrow_type_to`` or if the key is not present in dictionary.
    """
    if key not in dictionary:
        raise ValueError(f"{key} is not present in the Dictionary.")

    value = dictionary[key]
    if isinstance(value, narrow_type_to):
        return value
    else:
        raise ValueError(f"Provided key ({key}) value does not have correct type")


def narrow_dict_type_and_set_attribute(
    self: object,
    dictionary: dict,
    dictionary_key: str,
    attribute_name: str,
    narrow_type_to: type[T],
    func: Callable[[Any], Any] | None = None,
) -> None:
    """
    Checks a key exists in dictionary, verify its type and sets the corresponding attribute. Optionally, passes
    narrowed value to function prior to setting attribute. If key is not present in dictionary or
    dictionary[dictionary_key] has the wrong type, a ``ValueError`` is thrown.

    Args:
        self (object): The object to set attribute to dictionary[dictionary_key].
        dictionary (dict[str, Any]): A dictionary with string keys.
        dictionary_key (str): The key to check dictionary for.
        narrow_type_to (type[T]): The expected type of dictionary[key].

    Raises:
        ValueError: If dictionary[key] is not of type ``narrow_type_to`` or if the key is not present in dictionary.
    """
    val = narrow_dict_type(dictionary, dictionary_key, narrow_type_to)
    val = func(val) if func is not None else val
    setattr(self, attribute_name, val)


def make_dict_with_epochs_or_steps(local_epochs: int | None = None, local_steps: int | None = None) -> dict[str, int]:
    """
    Given two optional variables, this function will determine which, if any, are not None and create a dictionary
    from the value. If both are not None, it will prioritize ``local_epochs``. If both are None, then the dictionary
    is empty.

    Args:
        local_epochs (int | None, optional): Number of local epochs of training to perform in FL. Defaults to None.
        local_steps (int | None, optional): Number of local steps of training to perform in FL. Defaults to None.

    Returns:
        dict[str, int]: Dictionary with at most one of the non-none values, keyed by the name of the non-none variable
    """
    if local_epochs is not None:
        return {"local_epochs": local_epochs}
    if local_steps is not None:
        return {"local_steps": local_steps}
    else:
        return {}
