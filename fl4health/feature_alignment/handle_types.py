"""Largely taken from https://github.com/VectorInstitute/cyclops."""

from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_integer_dtype,
    is_numeric_dtype,
)

from fl4health.feature_alignment.constants import (
    FEATURE_INDICATOR_ATTR,
    FEATURE_MAPPING_ATTR,
    FEATURE_TYPE_ATTR,
    FEATURE_TYPES,
    FeatureType,
)


def _to_string(series: pd.Series) -> tuple[pd.Series, dict[str, Any]]:
    """
    Convert the features to string.

    Args:
        series (pd.Series): Feature data.

    Returns:
        (tuple[pd.Series, dict[str, Any]]): Tuple (pandas.Series, dict) with the updated feature data
        and metadata respectively.
    """
    convertible_to_type(series, FeatureType.STRING, unique=None, raise_error=True)
    return to_dtype(series, FeatureType.STRING), {FEATURE_TYPE_ATTR: FeatureType.STRING}


def _convertible_to_categorical_indicators(
    series: pd.Series,
    unique: np.ndarray | None = None,
    category_max: int = 20,
    raise_error_over_max: bool = False,
) -> bool:
    """
    Check whether a feature can be converted to categorical indicators.

    Args:
        series (pd.Series): Feature data.
        unique (np.ndarray | None, optional): Unique values which can be optionally specified. Defaults to None.
        category_max (int, optional): Maximum number of categories. Defaults to 20.
        raise_error_over_max (bool, optional): Whether to raise an error if categories exceeds max. Defaults to False.

    Returns:
        (bool):  Whether the feature can be converted.
    """
    return _convertible_to_categorical(
        series,
        category_min=2,
        category_max=category_max,
        unique=unique,
        raise_error_over_max=raise_error_over_max,
    )


def _to_categorical_indicators(
    data: pd.DataFrame, col: str, unique: np.ndarray | None = None
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Convert the features to binary categorical indicators.

    This performs the Pandas equivalent of one-hot encoding.

    Args:
        data (pd.DataFrame): Features data.
        col (str): Feature column being converted.
        unique (np.ndarray | None, optional): Unique values which can be optionally specified. Defaults to None.

    Raises:
        ValueError: Error if here are column duplicates.

    Returns:
        (tuple[pd.DataFrame, dict[str, Any]]): Tuple (pandas.DataFrame, dict) with the updated features data
        and metadata respectively.
    """
    series = data[col]
    unique = get_unique(series, unique=unique)
    dummies = pd.get_dummies(series, prefix=str(series.name))

    meta = {}
    for dummy_col in dummies.columns:
        dummies[dummy_col] = to_dtype(dummies[dummy_col], FeatureType.CATEGORICAL_INDICATOR)
        meta[dummy_col] = {
            FEATURE_TYPE_ATTR: FeatureType.CATEGORICAL_INDICATOR,
            FEATURE_INDICATOR_ATTR: col,
        }

    intersect = set(dummies.columns).intersection(data.columns)
    if len(intersect) > 0:
        raise ValueError(f"Cannot duplicate columns {', '.join(intersect)}.")

    data = pd.concat([data, dummies], axis=1)
    data = data.drop([col], axis=1)

    return data, meta


def _convertible_to_ordinal(
    series: pd.Series,
    unique: np.ndarray | None = None,
    category_max: int = 20,
    raise_error_over_max: bool = False,
) -> bool:
    """
    Check whether a feature can be converted to type ordinal.

    Args:
        series (pd.Series): Feature data.
        unique (np.ndarray | None, optional): Unique values which can be optionally specified. Defaults to None.
        category_max (int, optional): The number of categories above which the feature is not considered ordinal.
            Defaults to 20.
        raise_error_over_max (bool, optional): Whether to raise an error if there are more categories than max.
            Defaults to False.

    Returns:
        (bool): Whether the feature can be converted.
    """
    return _convertible_to_categorical(
        series,
        category_min=2,
        category_max=category_max,
        unique=unique,
        raise_error_over_max=raise_error_over_max,
    )


def _to_ordinal(series: pd.Series, unique: np.ndarray | None = None) -> tuple[pd.Series, dict[str, Any]]:
    """
    Convert the features to ordinal.

    Args:
        series (pd.Series): Feature data.
        unique (np.ndarray | None, optional): Unique values which can be optionally specified. Defaults to None.

    Returns:
        (tuple[pd.Series, dict[str, Any]]): Tuple (pandas.Series, dict) with the updated feature data
        and metadata respectively.
    """
    series, meta = _numeric_categorical_mapping(series, unique=unique)
    meta[FEATURE_TYPE_ATTR] = FeatureType.ORDINAL
    return to_dtype(series, FeatureType.ORDINAL), meta


def _numeric_categorical_mapping(
    series: pd.Series, unique: np.ndarray | None = None
) -> tuple[pd.Series, dict[str, Any]]:
    """
    Map values to categories in a series.

    Args:
        series (pd.Series): Feature data.
        unique (np.ndarray | None, optional): Unique values which can be optionally specified. Defaults to None.

    Returns:
        (tuple[pd.Series, dict[str, Any]]): Tuple (pandas.Series, dict) with the updated feature data and metadata
        respectively.
    """
    unique = get_unique(series, unique=unique)
    if unique.dtype.name == "object":
        unique = unique.astype(str)

    unique.sort()

    map_dict: dict[Any, int] = {}
    for i, unique_val in enumerate(unique):
        map_dict[unique_val] = i

    series = series.map(map_dict)

    inv_map = {v: k for k, v in map_dict.items()}
    meta = {FEATURE_MAPPING_ATTR: inv_map}

    return series, meta


def _convertible_to_binary(series: pd.Series, unique: np.ndarray | None = None) -> bool:
    """
    Check whether a feature can be converted to type binary.

    Args:
        series (pd.Series): Feature data.
        unique (np.ndarray | None, optional): Unique values which can be optionally specified. Defaults to None.

    Returns:
        (bool): Whether the feature can be converted.
    """
    if is_bool_dtype(series):
        return True

    return _convertible_to_categorical(
        series,
        category_min=2,
        category_max=2,
        unique=unique,
    )


def _to_binary(series: pd.Series, unique: np.ndarray | None = None) -> tuple[pd.Series, dict[str, Any]]:
    """
    Convert the features to binary.

    Args:
        series (pd.Series): Feature data.
        unique (np.ndarray | None, optional): Unique values which can be optionally specified. Defaults to None.

    Returns:
        (tuple[pd.Series, dict[str, Any]]): Tuple (pandas.Series, dict) with the updated feature data and metadata
        respectively.
    """
    if is_bool_dtype(series):
        meta = {
            FEATURE_TYPE_ATTR: FeatureType.BINARY,
            FEATURE_MAPPING_ATTR: {False: False, True: True},
        }
        return to_dtype(series, FeatureType.BINARY), meta

    series, meta = _numeric_categorical_mapping(series, unique=unique)
    meta[FEATURE_TYPE_ATTR] = FeatureType.BINARY
    return to_dtype(series, FeatureType.BINARY), meta


def _convertible_to_numeric(series: pd.Series, raise_error: bool = False) -> bool:
    """
    Check whether a feature can be converted to type numeric.

    Args:
        series (pd.Series): Feature data.
        raise_error (bool, optional): Whether to raise an error if the type cannot be converted. Defaults to False.

    Returns:
        (bool): Whether the feature can be converted.
    """
    if raise_error:
        pd.to_numeric(series)
        return True

    try:
        pd.to_numeric(series)
        can_convert = True
    except (ValueError, TypeError):
        can_convert = False

    return can_convert


def _to_numeric(series: pd.Series, unique: np.ndarray | None = None) -> tuple[pd.Series, dict[str, Any]]:
    """
    Convert the features to numeric.

    Args:
        series (pd.Series): Feature data.
        unique (np.ndarray | None, optional): Unique values which can be optionally specified. Defaults to None.

    Returns:
        (tuple[pd.Series, dict[str, Any]]): Tuple (pandas.Series, dict) with the updated feature data and metadata
        respectively.
    """
    convertible_to_type(series, FeatureType.NUMERIC, unique=unique, raise_error=True)
    series = pd.to_numeric(series)
    return to_dtype(series, FeatureType.NUMERIC), {FEATURE_TYPE_ATTR: FeatureType.NUMERIC}


def _convertible_to_categorical(
    series: pd.Series,
    category_min: int | None = None,
    category_max: int | None = None,
    unique: np.ndarray | None = None,
    raise_error_over_max: bool = False,
    raise_error_under_min: bool = False,
) -> bool:
    """
    Check whether a feature can be converted to some categorical type.

    Args:
        series (pd.Series): Feature data.
        category_min (int | None, optional):  The minimum number of categories allowed. Defaults to None.
        category_max (int | None, optional): The maximum number of categories allowed. Defaults to None.
        unique (np.ndarray | None, optional): Unique values which can be optionally specified. Defaults to None.
        raise_error_over_max (bool, optional):  Whether to raise an error if there are more categories than max.
            Defaults to False.
        raise_error_under_min (bool, optional): Whether to raise an error if there are less categories than min.
            Defaults to False.

    Raises:
        ValueError: Raise an error if there are more categories than max and ``raise_error_over_max`` is True
        ValueError: Raise an error if there are less categories than min and ``raise_error_under_min`` is True

    Returns:
        (bool): Whether the feature can be converted.
    """
    # If numeric, only allow conversion if an integer type
    if is_numeric_dtype(series) and not is_integer_dtype(series):
        return False

    unique = get_unique(series, unique=unique)
    nonnull_unique = unique[~pd.isnull(unique)]
    nunique = len(nonnull_unique)

    satisfies_minimum_condition = True if category_min is None else nunique >= category_min

    satisfies_maximum_condition = True if category_max is None else nunique <= category_max

    # Convertible
    if satisfies_minimum_condition and satisfies_maximum_condition:
        return True

    # Not convertible
    if (not satisfies_maximum_condition) and raise_error_over_max:
        raise ValueError(
            f"Should have at most {category_max} categories, but has {nunique}.",
        )

    if (not satisfies_minimum_condition) and raise_error_under_min:
        raise ValueError(
            f"Should have at least {category_min} categories, but has {nunique}.",
        )

    return False


def convertible_to_type(
    series: pd.Series, type: FeatureType, unique: np.ndarray | None = None, raise_error: bool = False
) -> bool:
    """
    Check whether a feature can be converted to some type.

    Args:
        series (pd.Series): Feature data.
        type (FeatureType): Feature type name to check for conversion.
        unique (np.ndarray | None, optional): _description_. Defaults to None.
        raise_error (bool, optional): Unique values which can be optionally specified. Defaults to False.

    Raises:
        ValueError: Supported type has no corresponding datatype
        ValueError: Cannot convert series to the provided type and ``raise_error`` is true.

    Returns:
        (bool): Whether the feature can be converted.
    """
    if type == FeatureType.NUMERIC:
        convertible = _convertible_to_numeric(series)

    elif type == FeatureType.STRING:
        convertible = True

    elif type == FeatureType.BINARY:
        convertible = _convertible_to_binary(series, unique=unique)

    elif type == FeatureType.ORDINAL:
        convertible = _convertible_to_ordinal(series, unique=unique)

    elif type == FeatureType.CATEGORICAL_INDICATOR:
        convertible = _convertible_to_categorical_indicators(series, unique=unique)

    elif valid_feature_type(type, raise_error=True):
        # Check first if the type is valid, if so, then it isn't supported here.
        raise ValueError("Supported type has no corresponding datatype.")

    if raise_error and not convertible:
        raise ValueError(f"Cannot convert series {series.name} to type {type}.")

    return convertible


def get_unique(values: np.ndarray | pd.Series, unique: np.ndarray | None = None) -> np.ndarray:
    """
    Get the unique values of pandas series.

    The utility of this function comes from checking whether the unique values have already been calculated. This
    function assumes that if the unique values are passed, they are correct.

    Args:
        values (np.ndarray | pd.Series): Values for which to get the unique values.
        unique (np.ndarray | None, optional): Unique values which can be optionally specified. Defaults to None.

    Returns:
        (np.ndarray): The unique values.
    """
    if unique is None:
        return np.array(values.unique())  # type: ignore

    return unique


def valid_feature_type(type: FeatureType, raise_error: bool = True) -> bool:
    """
    Check whether a feature type name is valid.

    Args:
        type (FeatureType): Feature type name.
        raise_error (bool, optional): Whether to raise an error is the type is invalid. Defaults to True.

    Raises:
        ValueError: Raise when the type is invalid and ``raise_error`` is True

    Returns:
        (bool): Whether the type is valid.
    """
    if type in FEATURE_TYPES:
        return True

    if raise_error:
        all_feature_types = ", ".join([types.value for types in FEATURE_TYPES])
        raise ValueError(f"Feature type '{type.value}' not in {all_feature_types}.")

    return False


def _type_to_dtype(type: FeatureType) -> str | None:
    """
    Get the Pandas datatype for a feature type name.

    Args:
        type (FeatureType.): Feature type name.

    Raises:
        ValueError: Supported type has no corresponding datatype.

    Returns:
        (str | None): The feature's Pandas datatype, or None if no data type conversion is desired.
    """
    if type == FeatureType.STRING:
        # If string, leave as is - the user can choose the specific length/type.
        return None

    if type == FeatureType.NUMERIC:
        # If numeric, leave as is - the user can choose the precision.
        return None

    if type in (FeatureType.BINARY, FeatureType.CATEGORICAL_INDICATOR, FeatureType.ORDINAL):
        return "category"

    # Check first if the type is valid, if so, then it isn't supported in this function.
    if valid_feature_type(type, raise_error=True):
        raise ValueError("Supported type has no corresponding datatype.")

    return None


def to_dtype(series: pd.Series, type: FeatureType) -> pd.Series:
    """
    Set the series datatype according to the feature type.

    Args:
        series (pd.Series): Feature data.
        type (FeatureType): Feature type name.

    Returns:
        (pd.Series): The feature with the corresponding datatype.
    """
    dtype = _type_to_dtype(type)

    if dtype is None:
        return series

    if series.dtype == dtype:
        return series

    return series.astype(dtype)  # type: ignore


def _infer_type(series: pd.Series, unique: np.ndarray | None = None) -> FeatureType:
    """
    Infer intended feature type and perform the relevant conversion.

    Args:
        series (pd.Series): Feature data.
        unique (np.ndarray | None, optional): Unique values which can be optionally specified. Defaults to None.

    Raises:
        ValueError: Could not infer type of series

    Returns:
        (str): Feature type name.
    """
    unique = get_unique(series, unique=unique)

    if convertible_to_type(series, FeatureType.BINARY, unique=unique):
        return FeatureType.BINARY

    if convertible_to_type(series, FeatureType.ORDINAL, unique=unique):
        return FeatureType.ORDINAL

    if convertible_to_type(series, FeatureType.NUMERIC, unique=unique):
        return FeatureType.NUMERIC

    if convertible_to_type(series, FeatureType.STRING, unique=unique):
        return FeatureType.STRING

    raise ValueError(f"Could not infer type of series '{series.name}'.")


def _to_type(
    data: pd.DataFrame, col: str, new_type: FeatureType, unique: np.ndarray | None = None
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Convert a feature to a given type.

    Args:
        data (pd.DataFrame): Features data.
        col (str): Column name for the feature being converted.
        new_type (FeatureType): Feature type name of type to which to convert.
        unique (np.ndarray | None, optional): Unique values which can be optionally specified. Defaults to None.

    Raises:
        ValueError: The features data must be passed to keyword argument 'data'.
        ValueError: Cannot convert to the new type.

    Returns:
        (tuple[pd.Series | pd.DataFrame, dict[str, Any]]): Tuple (pandas.Series or pandas.DataFrame, dict) with the
            updated features data and metadata respectively. If converting to categorical indicators, a DataFrame is
            returned, otherwise a Series is returned.
    """
    if data is None:
        raise ValueError(
            "The features data must be passed to keyword argument 'data'.",
        )

    if new_type == FeatureType.CATEGORICAL_INDICATOR:
        return _to_categorical_indicators(data, col, unique=unique)

    if new_type == FeatureType.STRING:
        series, meta = _to_string(data[col])

    elif new_type == FeatureType.ORDINAL:
        series, meta = _to_ordinal(data[col], unique=unique)

    elif new_type == FeatureType.BINARY:
        series, meta = _to_binary(data[col], unique=unique)

    elif new_type == FeatureType.NUMERIC:
        series, meta = _to_numeric(data[col], unique=unique)

    elif valid_feature_type(new_type, raise_error=True):
        # Check if an incorrect type was passed, otherwise
        # say that it isn't supported.
        raise ValueError(f"Cannot convert to type {new_type}.")

    data[col] = series
    meta = {str(series.name): meta}
    return data, meta


def infer_types(data: pd.DataFrame, features: list[str]) -> dict[str, FeatureType]:
    """
    Infer intended feature types and perform the relevant conversions.

    Args:
        data (pd.DataFrame): Feature data.
        features (list[str]): Features to consider.

    Returns:
        (dict[str, str]): Feature name to feature type dictionary.
    """
    new_types = {}
    for col in features:
        new_types[col] = _infer_type(data[col])

    return new_types


def to_types(data: pd.DataFrame, new_types: dict[str, FeatureType]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Convert features to given types.

    Args:
        data (pd.DataFrame): Features data.
        new_types (dict[str, str]): Map from the feature column name to its new type.

    Returns:
        (tuple[pd.DataFrame, dict[str, Any]]): Tuple of pandas.DataFrame and dict with the updated features data and
            metadata respectively.
    """
    meta = {}
    for col, new_type in new_types.items():
        data, fmeta = _to_type(data, col, new_type)
        meta.update(fmeta)

    return data, meta
