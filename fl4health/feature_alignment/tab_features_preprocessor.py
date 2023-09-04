import pandas as pd
from flwr.common.typing import NDArray
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from fl4health.feature_alignment.tab_features_info_encoder import TabFeaturesInfoEncoder


class TabularFeaturesPreprocessor:
    def __init__(self, tab_feature_encoder: TabFeaturesInfoEncoder) -> None:
        categories = tab_feature_encoder.get_categories_list()
        numeric_features = tab_feature_encoder.features_by_type("numeric")
        binary_features = tab_feature_encoder.features_by_type("binary")
        ordinal_features = tab_feature_encoder.features_by_type("ordinal")

        numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="mean")), ("scaler", MinMaxScaler())])

        binary_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent"))])

        categorical_transformer = Pipeline(
            steps=[("encoder", OneHotEncoder(handle_unknown="ignore", categories=categories))]
        )

        self.column_transformer = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("bin", binary_transformer, binary_features),
                ("cat", categorical_transformer, ordinal_features),
            ],
            remainder="drop",
        )

    def align_features(self, df: pd.DataFrame) -> NDArray:
        return self.column_transformer.fit_transform(df)
