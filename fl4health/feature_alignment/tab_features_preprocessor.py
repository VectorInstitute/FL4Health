from typing import Tuple

import pandas as pd
from flwr.common.typing import NDArray
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder

from fl4health.feature_alignment.tab_features_info_encoder import TabFeaturesInfoEncoder


class TabularFeaturesPreprocessor:
    def __init__(self, tab_feature_encoder: TabFeaturesInfoEncoder) -> None:
        categories = tab_feature_encoder.get_categories_list()
        numeric_features = tab_feature_encoder.features_by_type("numeric")
        binary_features = tab_feature_encoder.features_by_type("binary")
        ordinal_features = tab_feature_encoder.features_by_type("ordinal")
        string_features = tab_feature_encoder.features_by_type("string")

        numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="mean")), ("scaler", MinMaxScaler())])

        binary_transformer = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OrdinalEncoder())]
        )

        categorical_transformer = Pipeline(
            steps=[("encoder", OneHotEncoder(handle_unknown="ignore", categories=categories))]
        )

        string_transformer = Pipeline(steps=[("vectorizer", TfidfVectorizer())])

        self.data_column_transformer = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("bin", binary_transformer, binary_features),
                ("cat", categorical_transformer, ordinal_features),
                ("str_vectorizer", string_transformer, string_features),
            ],
            remainder="drop",
        )

        self.target_transformer = self.construct_target_transformer(tab_feature_encoder)
        self.target_column = tab_feature_encoder.get_target()

    def construct_target_transformer(self, tab_feature_encoder: TabFeaturesInfoEncoder) -> ColumnTransformer:
        target_type = tab_feature_encoder.get_target_type()
        target = tab_feature_encoder.get_target()
        target_categories = tab_feature_encoder.get_target_categories()
        if target_type == "numeric":
            numeric_transformer = Pipeline(
                steps=[("imputer", SimpleImputer(strategy="mean")), ("scaler", MinMaxScaler())]
            )
            return ColumnTransformer(transformers=[("num", numeric_transformer, [target])], remainder="drop")
        elif target_type == "binary":
            binary_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent"))])
            return ColumnTransformer(transformers=[("bin", binary_transformer, [target])], remainder="drop")
        elif target_type == "ordinal":
            categorical_transformer = Pipeline(
                steps=[("encoder", OneHotEncoder(handle_unknown="ignore", categories=target_categories))]
            )
            return ColumnTransformer(transformers=[("cat", categorical_transformer, [target])], remainder="drop")

    def preprocess_features(self, df: pd.DataFrame) -> Tuple[NDArray, NDArray]:

        return self.data_column_transformer.fit_transform(
            df.drop(columns=[self.target_column])
        ), self.target_transformer.fit_transform(df[[self.target_column]])
