import json
from typing import Dict, List, Optional, Union

import pandas as pd
from cyclops.process.feature.feature import TabularFeatures
from flwr.common.typing import Scalar
from sklearn.feature_extraction.text import CountVectorizer

from fl4health.feature_alignment.tabular_feature import MetaData, TabularFeature
from fl4health.feature_alignment.tabular_type import TabularType


class TabularFeaturesInfoEncoder:
    """
    This class encodes all the information required to perform feature
    alignment on tabular datasets.

    Args:
        tabular_features (List[TabularFeature]): List of all tabular features.
        tabular_targets (List[TabularFeature]): List of all targets.
        (Note: targets are not included in tabular_features)
    """

    def __init__(self, tabular_features: List[TabularFeature], tabular_targets: List[TabularFeature]) -> None:
        self.tabular_features = sorted(tabular_features, key=TabularFeature.get_feature_name)
        self.tabular_targets = sorted(tabular_targets, key=TabularFeature.get_feature_name)

    def get_tabular_features(self) -> List[TabularFeature]:
        return self.tabular_features

    def get_tabular_targets(self) -> List[TabularFeature]:
        return self.tabular_targets

    def get_feature_columns(self) -> List[str]:
        return sorted([feature.get_feature_name() for feature in self.tabular_features])

    def get_target_columns(self) -> List[str]:
        return sorted([target.get_feature_name() for target in self.tabular_targets])

    def features_by_type(self, tabular_type: TabularType) -> List[TabularFeature]:
        return sorted(
            [feature for feature in self.tabular_features if feature.get_feature_type() == tabular_type],
            key=TabularFeature.get_feature_name,
        )

    def type_to_features(self) -> Dict[TabularType, List[TabularFeature]]:
        return {tabular_type: self.features_by_type(tabular_type) for tabular_type in TabularType}

    def get_categories_list(self) -> List[MetaData]:
        return [cat_feature.get_metadata() for cat_feature in self.features_by_type(TabularType.ORDINAL)]

    def get_target_dimension(self) -> int:
        # Return the dimension of the target array *after* feature alignment is performed.
        dimension = 0
        for target in self.tabular_targets:
            dimension += target.get_metadata_dimension()
        return dimension

    @staticmethod
    def _construct_tab_feature(
        df: pd.DataFrame,
        feature_name: str,
        feature_type: TabularType,
        fill_values: Optional[Dict[str, Scalar]],
    ) -> TabularFeature:
        if fill_values is None or feature_name not in fill_values:
            fill_value = TabularType.get_default_fill_value(feature_type)
        else:
            fill_value = fill_values[feature_name]

        if feature_type == TabularType.ORDINAL or feature_type == TabularType.BINARY:
            # Extract categories information.
            feature_categories = sorted(df[feature_name].unique().tolist())
            return TabularFeature(feature_name, feature_type, fill_value, feature_categories)
        elif feature_type == TabularType.STRING:
            # Extract vocabulary from a string column of df.
            count_vectorizer = CountVectorizer()
            count_vectorizer.fit(df[feature_name])
            vocabulary = count_vectorizer.vocabulary_
            return TabularFeature(feature_name, feature_type, fill_value, vocabulary)
        else:
            return TabularFeature(feature_name, feature_type, fill_value)

    @staticmethod
    def encoder_from_dataframe(
        df: pd.DataFrame,
        id_column: str,
        target_columns: Union[str, List[str]],
        fill_values: Optional[Dict[str, Scalar]] = None,
    ) -> "TabularFeaturesInfoEncoder":
        features_list = sorted(df.columns.values.tolist())
        features_list.remove(id_column)
        # Leverage cyclops to perform type inference
        tab_features = TabularFeatures(
            data=df.reset_index(), features=features_list, by=id_column, targets=target_columns
        )
        features_to_types = tab_features.types

        tabular_targets = []
        tabular_features = []
        # Construct TabularFeature objects.
        for feature_name in features_to_types:
            feature_type = TabularType(features_to_types[feature_name])
            tabular_feature = TabularFeaturesInfoEncoder._construct_tab_feature(
                df, feature_name, feature_type, fill_values
            )
            if feature_name == target_columns or feature_name in target_columns:
                tabular_targets.append(tabular_feature)
            else:
                tabular_features.append(tabular_feature)
        return TabularFeaturesInfoEncoder(tabular_features, tabular_targets)

    def to_json(self) -> str:
        return json.dumps(
            {
                "tabular_features": json.dumps([tab_feature.to_json() for tab_feature in self.tabular_features]),
                "tabular_targets": json.dumps([tab_target.to_json() for tab_target in self.tabular_targets]),
            }
        )

    @staticmethod
    def from_json(json_str: str) -> "TabularFeaturesInfoEncoder":
        attributes = json.loads(json_str)
        return TabularFeaturesInfoEncoder(
            [TabularFeature.from_json(tab_str) for tab_str in json.loads(attributes["tabular_features"])],
            [TabularFeature.from_json(target_str) for target_str in json.loads(attributes["tabular_targets"])],
        )
