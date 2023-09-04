import json
from typing import Dict, List

import pandas as pd
from cyclops.process.feature.feature import TabularFeatures
from flwr.common.typing import Scalar


class TabFeaturesInfoEncoder:
    def __init__(self, features_to_types: Dict[str, str], categories: Dict[str, List[Scalar]]) -> None:
        self.features_to_types = features_to_types
        self.categories = categories

    def features_by_type(self, feature_type: str) -> List[str]:
        return sorted([feature for feature, t in self.features_to_types.items() if t == feature_type])

    def get_categories(self) -> Dict[str, List[Scalar]]:
        return self.categories

    def get_categories_list(self) -> List[List[Scalar]]:
        return [self.get_categories()[feature_name] for feature_name in self.features_by_type("ordinal")]

    def merge(self, encoder: "TabFeaturesInfoEncoder") -> "TabFeaturesInfoEncoder":
        common_features_to_types = {
            feature: self.features_to_types[feature]
            for feature in self.features_to_types.keys() & encoder.features_to_types.keys()
            if self.features_to_types[feature] == encoder.features_to_types[feature]
        }
        new_categories = {
            feature_name: list(
                set(encoder.get_categories()[feature_name]).union(set(self.get_categories()[feature_name]))
            )
            for feature_name in common_features_to_types
            if common_features_to_types[feature_name] == "ordinal"
        }
        return TabFeaturesInfoEncoder(common_features_to_types, new_categories)

    @staticmethod
    def encoder_from_dataframe(df: pd.DataFrame, id_column: str, target_column: str) -> "TabFeaturesInfoEncoder":
        features_list = sorted(df.columns.values.tolist())
        features_list.remove(id_column)
        features_list.remove(target_column)
        tab_features = TabularFeatures(
            data=df.reset_index(), features=features_list, by=id_column, targets=target_column
        )
        features_to_types = tab_features.types
        ordinal_features: List[str] = sorted(tab_features.features_by_type("ordinal"))
        categories = {
            ordinal_feature: sorted(df[ordinal_feature].unique().tolist()) for ordinal_feature in ordinal_features
        }

        return TabFeaturesInfoEncoder(features_to_types, categories)

    def to_json(self) -> str:
        return json.dumps(
            {
                "features_to_types": json.dumps(self.features_to_types),
                "categories": json.dumps(self.categories),
            }
        )

    @staticmethod
    def from_json(json_str: str) -> "TabFeaturesInfoEncoder":
        attributes = json.loads(json_str)
        return TabFeaturesInfoEncoder(
            json.loads(attributes["features_to_types"]), json.loads(attributes["categories"])
        )
