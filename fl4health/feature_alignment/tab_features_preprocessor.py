from typing import Dict, List, Tuple

import pandas as pd
from constants import BINARY, FEATURE_TYPES, NUMERIC, ORDINAL, STRING, TextFeatureTransformer
from flwr.common.typing import NDArray, Scalar
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder

from fl4health.feature_alignment.tab_features_info_encoder import TabFeaturesInfoEncoder


class TabularFeaturesPreprocessor:
    def __init__(self, tab_feature_encoder: TabFeaturesInfoEncoder) -> None:
        self.categories = tab_feature_encoder.get_categories_list()
        self.target_column = tab_feature_encoder.get_target()
        self.type_to_features: Dict[str, List[str]] = {
            feature_type: tab_feature_encoder.features_by_type(feature_type) for feature_type in FEATURE_TYPES
        }
        self.default_fill_values: Dict[str, Scalar] = tab_feature_encoder.get_all_default_fill_values()
        self.transformers: Dict[str, Pipeline] = {}

        self.transformers[NUMERIC] = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="mean")), ("scaler", MinMaxScaler())]
        )
        self.transformers[BINARY] = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OrdinalEncoder())]
        )
        self.transformers[ORDINAL] = Pipeline(
            steps=[("encoder", OneHotEncoder(handle_unknown="ignore", categories=self.categories))]
        )
        self.transformers[STRING] = Pipeline(steps=[("vectorizer", StringColumnTransformer(TfidfVectorizer()))])

        self.data_column_transformer = ColumnTransformer(
            transformers=[
                ("num", self.transformers[NUMERIC], self.type_to_features[NUMERIC]),
                ("bin", self.transformers[BINARY], self.type_to_features[BINARY]),
                ("str_vectorizer", self.transformers[STRING], self.type_to_features[STRING]),
                ("cat", self.transformers[ORDINAL], self.type_to_features[ORDINAL]),
            ],
            remainder="drop",
        )

        self.target_transformer = self.construct_target_transformer(tab_feature_encoder)

    def construct_target_transformer(self, tab_feature_encoder: TabFeaturesInfoEncoder) -> ColumnTransformer:
        # We assume that the target column is of type BINARY, NUMERIC, or ORDINAL.
        target_type = tab_feature_encoder.get_target_type()
        target = tab_feature_encoder.get_target()
        target_categories = tab_feature_encoder.get_target_categories()
        if target_type == NUMERIC:
            numeric_transformer = Pipeline(
                steps=[("imputer", SimpleImputer(strategy="mean")), ("scaler", MinMaxScaler())]
            )
            return ColumnTransformer(transformers=[("num", numeric_transformer, [target])], remainder="drop")
        elif target_type == BINARY:
            binary_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent"))])
            return ColumnTransformer(transformers=[("bin", binary_transformer, [target])], remainder="drop")
        elif target_type == ORDINAL:
            categorical_transformer = Pipeline(
                steps=[("encoder", OneHotEncoder(handle_unknown="ignore", categories=target_categories))]
            )
            return ColumnTransformer(transformers=[("cat", categorical_transformer, [target])], remainder="drop")

    def set_data_transformer(self, data_type: str, transformer: Pipeline) -> None:
        assert data_type in FEATURE_TYPES
        self.transformers[data_type] = transformer

    def set_target_transformer(self, transformer: Pipeline) -> None:
        self.target_transformer = transformer

    def preprocess_features(self, df: pd.DataFrame) -> Tuple[NDArray, NDArray]:
        # If the dataframe has an entire column missing, we need to fill it with some default value first.
        df_filled = self.fill_in_missing_columns(df)
        # After filling in missing columns, apply the feature alignment transform.
        return self.data_column_transformer.fit_transform(
            df_filled.drop(columns=[self.target_column])
        ), self.target_transformer.fit_transform(df_filled[[self.target_column]])

    def fill_in_missing_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        # Return a new DataFrame where entire missing columns are filled with values specified in
        # self.default_fill_values
        df_new = df.copy(deep=True)
        for feature_key, fill_in_val in self.default_fill_values.items():
            # fill in all columns of type "feature_key" with the correspondng default value, if missing.
            if feature_key in FEATURE_TYPES:
                for column_name in self.type_to_features[feature_key]:
                    self._fill_in_missing_column(df_new, column_name, fill_in_val)
            # Otherwise, feature_key is an actual column name instead of a type.
            else:
                self._fill_in_missing_column(df_new, feature_key, fill_in_val)
        return df_new

    def _fill_in_missing_column(self, df: pd.DataFrame, column_name: str, value: Scalar) -> None:
        if column_name not in df.columns:
            df[column_name] = value


class StringColumnTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, transformer: TextFeatureTransformer):
        self.transformer = transformer

    def fit(self, X: pd.DataFrame, y: pd.DataFrame = None) -> "StringColumnTransformer":
        joined_X = X.apply(lambda x: " ".join(x), axis=1)
        self.transformer.fit(joined_X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        joined_X = X.apply(lambda x: " ".join(x), axis=1)
        return self.transformer.transform(joined_X)
