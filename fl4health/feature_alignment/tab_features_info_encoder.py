import json
from typing import Dict, List

import pandas as pd
from cyclops.process.feature.feature import TabularFeatures
from flwr.common.typing import Scalar
from sklearn.feature_extraction.text import CountVectorizer

from fl4health.feature_alignment.constants import BINARY, DEFAULT_FILL_VALUES, ORDINAL, STRING
from fl4health.feature_alignment.string_columns_transformer import StringColumnTransformer


class TargetInfoEncoder:
    """
    This class encodes the information about the target column(s)
    that is necessary to perform feature alignment.

    Parameters
    ----------
    target:
    target_type:
    target_categories:
    """

    def __init__(self, target: str, target_type: str, target_categories: List[Scalar]) -> None:
        self.target = target
        self.target_type = target_type
        self.target_categories = target_categories

    def get_target(self) -> str:
        return self.target

    def get_target_type(self) -> str:
        return self.target_type

    def get_target_categories(self) -> List[Scalar]:
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
    """
    This class encodes all the information required to perform feature
    alignment on tabular datasets.

    Parameters
    ----------
    features_to_types: Dict[str, str]
        Dictionary that maps each feature name to its type.
        We consider four types in tabular data:
        BINARY, ORDINAL, NUMERICAL, and STRING.
    categories: Dict[str, List[Scalar]]
        Dictionary that maps each ordinal feature to its categories.
    target_info: TargetInfoEncoder
        Information about the target column(s).
    vocabulary: Dict[str, int]
        Vocabulary of all the STRING columns.
    default_fill_values: Dict[str, Scalar]
        The default values used to fill in missing values.
        Each of the four types has a default fill-in value,
        but each specific feature can also has its own default fill-in value.
    """

    def __init__(
        self,
        features_to_types: Dict[str, str],
        categories: Dict[str, List[Scalar]],
        target_info: TargetInfoEncoder,
        vocabulary: Dict[str, int],
        default_fill_values: Dict[str, Scalar] = DEFAULT_FILL_VALUES,
    ) -> None:
        self.features_to_types = features_to_types
        self.categories = categories
        self.target_info = target_info
        self.vocabulary = vocabulary
        self.default_fill_values = default_fill_values

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

    def get_target_categories(self) -> List[Scalar]:
        return self.target_info.get_target_categories()

    def get_vocabulary(self) -> Dict[str, int]:
        return self.vocabulary

    def get_all_default_fill_values(self) -> Dict[str, Scalar]:
        return self.default_fill_values

    def get_default_fill_value(self, key: str) -> Scalar:
        return self.default_fill_values[key]

    def set_default_fill_value(self, key: str, value: Scalar) -> None:
        self.default_fill_values[key] = value

    # def merge(self, encoder: "TabFeaturesInfoEncoder") -> "TabFeaturesInfoEncoder":
    #     assert self.get_target() == encoder.get_target() and self.get_target_type() == encoder.get_target_type()
    #     common_features_to_types = {
    #         feature: self.features_to_types[feature]
    #         for feature in self.features_to_types.keys() & encoder.features_to_types.keys()
    #         if self.features_to_types[feature] == encoder.features_to_types[feature]
    #     }
    #     new_categories = {
    #         feature_name: list(
    #             set(encoder.get_categories()[feature_name]).union(set(self.get_categories()[feature_name]))
    #         )
    #         for feature_name in common_features_to_types
    #         if common_features_to_types[feature_name] == "ordinal"
    #     }

    #     new_target_info = TargetInfoEncoder(
    #         self.get_target(),
    #         self.get_target_type(),
    #         list(set(self.get_target_categories()).union(encoder.get_target_categories())),
    #     )
    #     return TabFeaturesInfoEncoder(common_features_to_types, new_categories, new_target_info)

    @staticmethod
    def encoder_from_dataframe(
        df: pd.DataFrame,
        id_column: str,
        target_column: str,
        default_fill_values: Dict[str, Scalar] = DEFAULT_FILL_VALUES,
    ) -> "TabFeaturesInfoEncoder":
        features_list = sorted(df.columns.values.tolist())
        features_list.remove(id_column)
        tab_features = TabularFeatures(
            data=df.reset_index(), features=features_list, by=id_column, targets=target_column
        )
        features_to_types = tab_features.types
        target_type = features_to_types[target_column]

        # The target column is separated from the other columns
        # so that targets and features remain separate after alignment
        features_to_types.pop(target_column)

        # extract categories information
        ordinal_features: List[str] = sorted(tab_features.features_by_type("ordinal"))
        string_features: List[str] = sorted(tab_features.features_by_type(type_=STRING))

        categories = {
            ordinal_feature: sorted(df[ordinal_feature].unique().tolist()) for ordinal_feature in ordinal_features
        }

        if target_type == ORDINAL or target_type == BINARY:
            target_categories = sorted(df[target_column].unique().tolist())
        else:
            target_categories = []

        target_info = TargetInfoEncoder(target_column, target_type, target_categories)

        # Extract vocabulary from the string columns of df
        count_vectorizer = CountVectorizer()
        string_col_transformer = StringColumnTransformer(count_vectorizer)
        string_col_transformer.fit(df[string_features])
        vocabulary = count_vectorizer.vocabulary_

        return TabFeaturesInfoEncoder(features_to_types, categories, target_info, vocabulary, default_fill_values)

    def to_json(self) -> str:
        return json.dumps(
            {
                "features_to_types": json.dumps(self.features_to_types),
                "categories": json.dumps(self.categories),
                "target_info": json.dumps(self.target_info.to_json()),
                "vocabulary": json.dumps(self.vocabulary),
                "default_fill_values": json.dumps(self.default_fill_values),
            }
        )

    @staticmethod
    def from_json(json_str: str) -> "TabFeaturesInfoEncoder":
        attributes = json.loads(json_str)
        return TabFeaturesInfoEncoder(
            json.loads(attributes["features_to_types"]),
            json.loads(attributes["categories"]),
            TargetInfoEncoder.from_json(json.loads(attributes["target_info"])),
            json.loads(attributes["vocabulary"]),
            json.loads(attributes["default_fill_values"]),
        )
