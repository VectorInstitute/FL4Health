"""Largely taken from https://github.com/VectorInstitute/cyclops."""

from typing import Any

import numpy as np
import pandas as pd

from fl4health.feature_alignment.constants import (
    FEATURE_INDICATOR_ATTR,
    FEATURE_META_ATTR_DEFAULTS,
    FEATURE_META_ATTRS,
    FEATURE_TYPE_ATTR,
    FEATURE_TYPES,
)
from fl4health.feature_alignment.handle_types import FeatureType, infer_types, to_types


def to_list(obj: Any) -> list[Any]:
    """
    Convert some object to a list of object(s) unless already one.

    Args:
        obj (Any): The object to convert to a list.

    Returns:
        (list[Any]): The processed object.
    """
    if isinstance(obj, list):
        return obj

    if isinstance(obj, (np.ndarray, set, dict)):
        return list(obj)

    return [obj]


def has_columns(data: pd.DataFrame, cols: str | list[str], exactly: bool = False, raise_error: bool = False) -> bool:
    """
    Check if data has required columns for processing.

    Args:
        data (pd.DataFrame): DataFrame to check.
        cols (str | list[str]): List of column names that must be present in data.
        exactly (bool, optional): Whether columns need to be an exact match. Defaults to False.
        raise_error (bool, optional): Whether to raise a ValueError if there are missing columns. Defaults to False.

    Raises:
        ValueError: Missing required columns.
        ValueError: Must have exactly the columns, will throw if not and exactly is True.

    Returns:
        (bool): True if all required columns are present, otherwise False.
    """
    cols = to_list(cols)
    required_set = set(cols)
    columns = set(data.columns)
    present = required_set.issubset(columns)

    if not present and raise_error:
        missing = required_set - columns
        raise ValueError(f"Missing required columns: {', '.join(missing)}.")

    if exactly:
        exact = present and len(data.columns) == len(cols)
        if not exact and raise_error:
            raise ValueError(f"Must have exactly the columns: {', '.join(cols)}.")

    return present


class FeatureMeta:
    def __init__(self, **kwargs: Any) -> None:
        """Feature metadata class."""
        # Feature type checking
        if FEATURE_TYPE_ATTR not in kwargs:
            raise ValueError("Must specify feature type.")

        if kwargs[FEATURE_TYPE_ATTR] not in FEATURE_TYPES:
            all_feature_types = ", ".join([types.value for types in FEATURE_TYPES])
            raise ValueError(
                f"Feature type '{kwargs[FEATURE_TYPE_ATTR]}'\nnot in {all_feature_types}.",
            )

        # Set attributes
        for attr in FEATURE_META_ATTRS:
            if attr in kwargs:
                setattr(self, attr, kwargs[attr])
            else:
                setattr(self, attr, FEATURE_META_ATTR_DEFAULTS[attr])

        # Check for invalid parameters
        invalid_params = [kwarg for kwarg in kwargs if kwarg not in FEATURE_META_ATTRS]
        if len(invalid_params) > 0:
            raise ValueError(
                f"Invalid feature meta parameters {', '.join(invalid_params)}.",
            )

    def get_type(self) -> FeatureType:
        """
        Get the feature type.

        Returns:
            (str):  Feature type.
        """
        return FeatureType(getattr(self, FEATURE_TYPE_ATTR))

    def update(self, meta: list[tuple[str, Any]]) -> None:
        """
        Update meta attributes.

        Args:
            meta (list[tuple[str, Any]]): List of tuples in the format (attribute name, attribute value).
        """
        for info in meta:
            setattr(self, *info)


class Features:
    def __init__(
        self,
        data: pd.DataFrame,
        features: str | list[str],
        by: str | list[str] | None = None,
        targets: str | list[str] | None = None,
        force_types: dict[str, FeatureType] | None = None,
    ):
        """
        Features.

        Args:
            data (pd.DataFrame): Features data.
            features (str | list[str]): List of feature columns. The remaining columns are treated as metadata.
            by (str | list[str] | None, optional): Columns to groupby during processing, affecting how the features
                are treated. Defaults to None.
            targets (str | list[str] | None, optional): Column names to specify as target features. Defaults to None.
            force_types (dict[str, FeatureType] | None, optional): Mapping of column names to type. These columns are
                forced to be of the specified type. Defaults to None.
        """
        # Check data
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Feature data must be a pandas.DataFrame.")

        target_list = [] if targets is None else to_list(targets)
        feature_list = to_list(features)
        if len(feature_list) == 0:
            raise ValueError("Must specify at least one feature.")

        has_columns(data, feature_list, raise_error=True)
        has_columns(data, target_list, raise_error=True)

        self.by_ = [] if by is None else to_list(by)
        if len(self.by_) > 0:
            has_columns(data, self.by_, raise_error=True)
            if len(set(self.by_).intersection(set(feature_list))) != 0:
                raise ValueError("Columns in 'by' cannot be considered features.")

        # Add targets to the list of features if they were not included
        self.features = list(set(feature_list + target_list))
        self.data = data
        self.meta: dict[str, FeatureMeta] = {}
        self._infer_feature_types(force_types=force_types)

    @property
    def types(self) -> dict[str, FeatureType]:
        """
        Access as attribute, feature type names.

        NOTE: These are framework-specific feature names.

        Returns:
            (dict[str, str]): Feature type mapped for each feature.
        """
        return {name: meta.get_type() for name, meta in self.meta.items()}

    def _update_meta(self, meta_update: dict[str, dict[str, Any]]) -> None:
        """
        Update feature metadata.

        Args:
            meta_update (dict[str, dict[str, Any]]): A dictionary in which the values will add/update
                the existing feature metadata dictionary.
        """
        for col, info in meta_update.items():
            if col in self.meta:
                self.meta[col].update(list(info.items()))
            else:
                self.meta[col] = FeatureMeta(**info)

    def _to_feature_types(
        self, data: pd.DataFrame, new_types: dict[str, FeatureType], inplace: bool = True
    ) -> pd.DataFrame:
        """
        Convert feature types.

        Args:
            data (pd.DataFrame): Features data.
            new_types (dict[str, FeatureType]): A map from the feature name to the new feature type.
            inplace (bool, optional): Whether to perform in-place, or to simply return the DataFrame. Defaults to True.

        Raises:
            ValueError: Unrecognized features
            ValueError: When conversion fails

        Returns:
            (pd.DataFrame): The features data with the relevant conversions.
        """
        invalid = set(new_types.keys()) - set(self.features)
        if len(invalid) > 0:
            raise ValueError(f"Unrecognized features: {', '.join(invalid)}")
        for col, new_type in new_types.items():
            if col in self.meta and inplace and new_type == FeatureType.CATEGORICAL_INDICATOR:
                raise ValueError(
                    f"Cannot convert {col} to binary categorical indicators.",
                )
        data, meta = to_types(data, new_types)
        if inplace:
            # Append any new indicator features
            for col, fmeta in meta.items():
                if FEATURE_INDICATOR_ATTR in fmeta:
                    self.features.append(col)

            self.data = data
            self._update_meta(meta)

        return data

    def _infer_feature_types(self, force_types: dict[str, FeatureType] | None = None) -> None:
        """
        Infer feature types. Can optionally force certain types on specified features.

        Args:
            force_types (dict[str, FeatureType] | None, optional): A map from the feature name to the forced feature
                type. Defaults to None.
        """
        if force_types is None:
            force_types = {}
        infer_features = to_list(
            set(self.features) - set(to_list(force_types)),
        )
        new_types = infer_types(self.data, infer_features)
        # Force certain features to be specific types
        if force_types is not None:
            for feature, type_ in force_types.items():
                new_types[feature] = type_

        self._to_feature_types(self.data, new_types)


class TabularFeatures(Features):
    def __init__(
        self,
        data: pd.DataFrame,
        features: str | list[str],
        by: str,
        targets: str | list[str] | None = None,
        force_types: dict[str, FeatureType] | None = None,
    ):
        """
        Tabular features.

        Args:
            data (pd.DataFrame): Data for the table
            features (str | list[str]): List of feature columns. The remaining columns are treated as metadata.
            by (str): Columns to groupby during processing, affecting how the features are treated.
            targets (str | list[str] | None, optional): Column names to specify as target features.
                Defaults to None.
            force_types (dict[str, FeatureType] | None, optional): Mapping of column names to type. These columns are
                forced to be of the specified type. Defaults to None.

        Raises:
            ValueError: Tabular features index input as a string representing a column
        """
        if not isinstance(by, str):
            raise ValueError(
                "Tabular features index input as a string representing a column.",
            )

        super().__init__(
            data,
            features,
            by,
            targets=targets,
            force_types=force_types,
        )
