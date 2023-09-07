import json
from typing import Dict, List

import pandas as pd
from cyclops.process.feature.feature import TabularFeatures
from flwr.common.typing import Scalar


class TargetInfoEncoder:
    def __init__(self, target: str, target_type: str, target_categories: List[str]) -> None:
        self.target = target
        self.target_type = target_type
        self.target_categories = target_categories

    def get_target(self) -> str:
        return self.target

    def get_target_type(self) -> str:
        return self.target_type

    def get_target_categories(self) -> List[str]:
        return self.target_categories

    def to_json(self) -> str:
        return json.dumps(
            {
                "target": json.dumps(self.get_target()),
                "target_type": json.dumps(self.get_target_type()),
                "target_categories": json.dumps(self.get_target_categories()),
            }
        )

    @staticmethod
    def from_json(json_str: str) -> "TargetInfoEncoder":
        attributes = json.loads(json_str)
        return TargetInfoEncoder(
            json.loads(attributes["target"]),
            json.loads(attributes["target_type"]),
            json.loads(attributes["target_categories"]),
        )


class TabFeaturesInfoEncoder:
    def __init__(
        self, features_to_types: Dict[str, str], categories: Dict[str, List[Scalar]], target_info: TargetInfoEncoder
    ) -> None:
        self.features_to_types = features_to_types
        self.categories = categories
        self.target_info = target_info

    def features_by_type(self, feature_type: str) -> List[str]:
        return sorted([feature for feature, t in self.features_to_types.items() if t == feature_type])

    def get_categories(self) -> Dict[str, List[Scalar]]:
        return self.categories

    def get_categories_list(self) -> List[List[Scalar]]:
        return [self.get_categories()[feature_name] for feature_name in self.features_by_type("ordinal")]

    def get_target(self) -> str:
        return self.target_info.get_target()

    def get_target_type(self) -> str:
        return self.target_info.get_target_type()

    def get_target_categories(self) -> List[str]:
        return self.target_info.get_target_categories()

    def merge(self, encoder: "TabFeaturesInfoEncoder") -> "TabFeaturesInfoEncoder":
        assert self.get_target() == encoder.get_target() and self.get_target_type() == encoder.get_target_type()
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

        new_target_info = TargetInfoEncoder(
            self.get_target(),
            self.get_target_type(),
            list(set(self.get_target_categories()).union(encoder.get_target_categories())),
        )
        return TabFeaturesInfoEncoder(common_features_to_types, new_categories, new_target_info)

    @staticmethod
    def encoder_from_dataframe(df: pd.DataFrame, id_column: str, target_column: str) -> "TabFeaturesInfoEncoder":
        features_list = sorted(df.columns.values.tolist())
        features_list.remove(id_column)
        # features_list.remove(target_column)
        tab_features = TabularFeatures(
            data=df.reset_index(), features=features_list, by=id_column, targets=target_column
        )
        features_to_types = tab_features.types
        target_type = features_to_types[target_column]
        features_to_types.pop(target_column)
        ordinal_features: List[str] = sorted(tab_features.features_by_type("ordinal"))
        categories = {
            ordinal_feature: sorted(df[ordinal_feature].unique().tolist()) for ordinal_feature in ordinal_features
        }

        if target_type == "ordinal":
            target_categories = sorted(df[target_column].unique().tolist())
        else:
            target_categories = []

        target_info = TargetInfoEncoder(target_column, target_type, target_categories)

        return TabFeaturesInfoEncoder(features_to_types, categories, target_info)

    def to_json(self) -> str:
        return json.dumps(
            {
                "features_to_types": json.dumps(self.features_to_types),
                "categories": json.dumps(self.categories),
                "target_info": json.dumps(self.target_info.to_json()),
            }
        )

    @staticmethod
    def from_json(json_str: str) -> "TabFeaturesInfoEncoder":
        attributes = json.loads(json_str)
        return TabFeaturesInfoEncoder(
            json.loads(attributes["features_to_types"]),
            json.loads(attributes["categories"]),
            TargetInfoEncoder.from_json(attributes["target_info"]),
        )
