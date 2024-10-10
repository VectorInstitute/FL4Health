from typing import Any, Optional

from pytest import approx

DEFAULT_TOLERANCE = 0.0005


def assert_metrics_dict(metrics_to_assert: dict[str, Any], metrics_saved: dict[str, Any]) -> list[str]:
    errors = []

    def _assert(value: Any, saved_value: Any) -> Optional[str]:
        # helper function to avoid code repetition
        tolerance = DEFAULT_TOLERANCE
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

    for metric_key in metrics_to_assert:
        if metric_key not in metrics_saved:
            errors.append(f"Metric '{metric_key}' not found in saved metrics.")
            continue

        value_to_assert = metrics_to_assert[metric_key]

        if isinstance(value_to_assert, dict):
            if "target_value" not in value_to_assert and "custom_tolerance" not in value_to_assert:
                # if it's a dictionary, call this function recursively
                # except when the dictionary has "target_value" and "custom_tolerance", which should
                # be treated as a regular dictionary
                errors.extend(assert_metrics_dict(value_to_assert, metrics_saved[metric_key]))
                continue

        if isinstance(value_to_assert, list) and len(value_to_assert) > 0:
            # if it's a list, call an assertion for each element of the list
            for i in range(len(value_to_assert)):
                error = _assert(value_to_assert[i], metrics_saved[metric_key][i])
                if error is not None:
                    errors.append(error)
            continue

        # if it's just a regular value, perform the assertion
        error = _assert(value_to_assert, metrics_saved[metric_key])
        if error is not None:
            errors.append(error)

    return errors
