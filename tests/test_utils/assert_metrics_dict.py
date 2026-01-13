from typing import Any

from pytest import approx


DEFAULT_TOLERANCE = 0.0005


def _assert(value: Any, saved_value: Any, metric_key: str, tolerance: float = DEFAULT_TOLERANCE) -> str | None:
    # helper function to avoid code repetition
    if isinstance(value, dict):
        # if the value is a dictionary, extract the target value and the custom tolerance
        tolerance = value["custom_tolerance"]
        value = value["target_value"]

    if approx(value, abs=tolerance) != saved_value:
        return (
            f"Saved value for metric '{metric_key}' ({saved_value}) does not match the requested "
            f"value ({value}) within requested tolerance ({tolerance})."
        )

    return None


def assert_metrics_dict(metrics_to_assert: dict[str, Any], metrics_saved: dict[str, Any]) -> list[str]:
    """
    Recursively compares two dictionaries to ensure the values are the same.

    Ensures that the key value pairs in 'metrics_to_assert' are present and within the requested tolerances in
    'metrics_saved'.

    Args:
        metrics_to_assert (dict[str, Any]): A dictionary containing metrics, or any other FL experiment outputs which
            are considered to be the 'ground truth' values which we are testing against. This dictionary can be a
            subset of the 'metrics_saved' dictionary if only certain key value pairs are to be tested.
        metrics_saved (dict[str, Any]): A dictionary containing metrics or any other FL experiment outputs which are
            to be tested.

    Returns:
        (list[str]): A list of error messages. If the assertion passes then this list will be empty.
    """
    errors = []

    for metric_key, value_to_assert in metrics_to_assert.items():
        if metric_key not in metrics_saved:
            errors.append(f"Metric '{metric_key}' not found in saved metrics.")
            continue

        if (
            isinstance(value_to_assert, dict)
            and "target_value" not in value_to_assert
            and "custom_tolerance" not in value_to_assert
        ):
            # if it's a dictionary, call this function recursively
            # except when the dictionary has "target_value" and "custom_tolerance", which should
            # be treated as a regular dictionary
            errors.extend(assert_metrics_dict(value_to_assert, metrics_saved[metric_key]))
            continue

        if isinstance(value_to_assert, list) and len(value_to_assert) > 0:
            # if it's a list, call an assertion for each element of the list
            for i in range(len(value_to_assert)):
                error = _assert(value_to_assert[i], metrics_saved[metric_key][i], metric_key)
                if error is not None:
                    errors.append(error)
            continue

        # if it's just a regular value, perform the assertion
        error = _assert(value_to_assert, metrics_saved[metric_key], metric_key)
        if error is not None:
            errors.append(error)

    return errors
