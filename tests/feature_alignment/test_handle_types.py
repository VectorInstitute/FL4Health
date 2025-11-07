import pandas as pd
import pytest

from fl4health.feature_alignment.constants import (
    BINARY,
    CATEGORICAL_INDICATOR,
    FEATURE_TYPE_ATTR,
    FEATURE_TYPES,
    STRING,
)
from fl4health.feature_alignment.handle_types import _to_string, _to_type, convertible_to_type


def test_to_string() -> None:
    series = pd.Series([1, 2, 3, 4, 5])
    converted_series, meta_data = _to_string(series)
    # Want to leave it alone and have user convert explicitly
    assert converted_series.dtype == "int64"
    assert meta_data[FEATURE_TYPE_ATTR] == STRING


def test_convertible_to_type() -> None:
    series = pd.Series([1, 2, 3, 4, 5])
    assert convertible_to_type(series, type=STRING)

    assert not convertible_to_type(series, type=BINARY)

    with pytest.raises(ValueError):
        convertible_to_type(series, type=BINARY, raise_error=True)

    assert convertible_to_type(series, type=CATEGORICAL_INDICATOR)

    # Too many categories
    with pytest.raises(ValueError):
        convertible_to_type(pd.Series(list(range(50))), type=CATEGORICAL_INDICATOR, raise_error=True)

    # Too few categories
    with pytest.raises(ValueError):
        convertible_to_type(pd.Series([0, 0, 0, 0]), type=CATEGORICAL_INDICATOR, raise_error=True)

    # Test with out of domain type
    with pytest.raises(ValueError):
        convertible_to_type(series, type="XYZ")

    # Temporarily Patch FEATURE_TYPES to exercise code line
    FEATURE_TYPES.append("XYZ")
    with pytest.raises(ValueError):
        convertible_to_type(series, type="XYZ")


def test_to_type() -> None:
    with pytest.raises(ValueError):
        _to_type(None, "", STRING)  # type: ignore
    cats = list(range(4)) + list(range(4)) + list(range(2))
    data = pd.DataFrame({"string_column": list(range(10)), "cat_column": cats})

    new_data, _ = _to_type(data, "string_column", STRING)
    assert new_data["string_column"].equals(data["string_column"])

    new_data, _ = _to_type(data, "cat_column", CATEGORICAL_INDICATOR)

    assert new_data["cat_column_0"].equals(
        pd.Series([True, False, False, False, True, False, False, False, True, False], index=list(range(10))).astype(
            "category"
        )
    )
