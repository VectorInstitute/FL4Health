from typing import Dict, List, Tuple

import pandas as pd
from flwr.common.typing import NDArray, Scalar
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder

from fl4health.feature_alignment.constants import BINARY, FEATURE_TYPES, NUMERIC, ORDINAL, STRING
from fl4health.feature_alignment.string_columns_transformer import StringColumnTransformer
from fl4health.feature_alignment.tab_features_info_encoder import TabularFeaturesInfoEncoder


class TabularFeaturesPreprocessor:
    """
    TabularFeaturesPreprocessor is responsible for constructing
    the appropriate column transformers based on the information
    encoded in tab_feature_encoder. These transformers will
    then be applied to a pandas dataframe.

    For each of the four feature types (BINARY, NUMERIC, ORDINAL, STRING),
    a default ColumnTransformer is constructed that is responsible for processing all
    columns of that type.

    Parameters
    ----------
    tab_feature_encoder: TabularFeaturesInfoEncoder
        Encodes the information necessary for constructing the column transformers.
    """

    def __init__(self, tab_feature_encoder: TabularFeaturesInfoEncoder) -> None:
        self.transformers: Dict[str, Pipeline] = {}
        self.initialize_default_data_transformer(tab_feature_encoder)
        self.initialize_default_target_transformer(tab_feature_encoder)

    def initialize_default_target_transformer(self, tab_feature_encoder: TabularFeaturesInfoEncoder) -> None:
        """
        Initialize a default column transformer for the target column.
        Assumption: the target column has type BINARY, NUMERIC, or ORDINAL.
        """
        target_type = tab_feature_encoder.get_target_type()
        target = tab_feature_encoder.get_target()
        target_categories = tab_feature_encoder.get_target_categories()
        if target_type == NUMERIC:
            numeric_transformer = Pipeline(
                steps=[("imputer", SimpleImputer(strategy="mean")), ("scaler", MinMaxScaler())]
            )
            self.target_column_transformer = ColumnTransformer(
                transformers=[("num", numeric_transformer, [target])], remainder="drop"
            )
        elif target_type == BINARY:
            binary_transformer = Pipeline(
                steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OrdinalEncoder())]
            )
            self.target_column_transformer = ColumnTransformer(
                transformers=[("bin", binary_transformer, [target])], remainder="drop"
            )
        elif target_type == ORDINAL:
            categorical_transformer = Pipeline(
                steps=[
                    (
                        "encoder",
                        OrdinalEncoder(
                            unknown_value=len(target_categories) + 1,
                            handle_unknown="use_encoded_value",
                            categories=[target_categories],
                        ),
                    )
                ]
            )
            self.target_column_transformer = ColumnTransformer(
                transformers=[("cat", categorical_transformer, [target])], remainder="drop"
            )

    def initialize_default_data_transformer(self, tab_feature_encoder: TabularFeaturesInfoEncoder) -> None:
        """
        Initialize a default ColumnTransformer for the data columns
        (i.e., all columns except the target column)
        """
        self.categories = tab_feature_encoder.get_categories_list()
        self.target_column = tab_feature_encoder.get_target()
        self.type_to_features: Dict[str, List[str]] = tab_feature_encoder.type_to_features()
        self.default_fill_values: Dict[str, Scalar] = tab_feature_encoder.get_all_default_fill_values()
        self.vocabulary = tab_feature_encoder.get_vocabulary()

        self.transformers[NUMERIC] = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="mean")), ("scaler", MinMaxScaler())]
        )
        self.transformers[BINARY] = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OrdinalEncoder())]
        )
        self.transformers[ORDINAL] = Pipeline(
            steps=[("encoder", OneHotEncoder(handle_unknown="ignore", categories=self.categories))]
        )
        self.transformers[STRING] = Pipeline(
            steps=[("vectorizer", StringColumnTransformer(TfidfVectorizer(vocabulary=self.vocabulary)))]
        )

        self.data_column_transformer = ColumnTransformer(
            transformers=[
                ("num", self.transformers[NUMERIC], self.type_to_features[NUMERIC]),
                ("bin", self.transformers[BINARY], self.type_to_features[BINARY]),
                ("str_vectorizer", self.transformers[STRING], self.type_to_features[STRING]),
                ("cat", self.transformers[ORDINAL], self.type_to_features[ORDINAL]),
            ],
            remainder="drop",
        )

    def set_data_transformer(self, data_type: str, transformer: Pipeline) -> None:
        """
        Allow the user to set a custom ColumnTransformer to be applied
        to features with type data_type.
        """
        assert data_type in FEATURE_TYPES
        self.transformers[data_type] = transformer

    def set_target_transformer(self, transformer: Pipeline) -> None:
        """
        Allow the user to set a custom ColumnTransformer for the target column.
        """
        self.target_column_transformer = transformer

    def preprocess_features(self, df: pd.DataFrame) -> Tuple[NDArray, NDArray]:
        """
        Apply self.data_column_transformer to all data columns
        and self.target_column_transformer to the target column to achieve
        feature alignment.
        """
        # If the dataframe has an entire column missing, we need to fill it with some default value first.
        df_filled = self.fill_in_missing_columns(df)
        # After filling in missing columns, apply the feature alignment transform.
        return self.data_column_transformer.fit_transform(
            df_filled.drop(columns=[self.target_column])
        ), self.target_column_transformer.fit_transform(df_filled[[self.target_column]])

    def fill_in_missing_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return a new DataFrame where entire missing columns are filled with values specified in
        self.default_fill_values
        """
        df_new = df.copy(deep=True)
        for feature_key, fill_in_val in self.default_fill_values.items():
            # fill in all columns of type "feature_key" with the correspondng default value, if missing.
            if feature_key in FEATURE_TYPES:
                for column_name in self.type_to_features[feature_key]:
                    if column_name not in self.default_fill_values.keys():
                        self._fill_in_missing_column(df_new, column_name, fill_in_val)
            # Otherwise, feature_key is an actual column name instead of a type.
            else:
                self._fill_in_missing_column(df_new, feature_key, fill_in_val)
        return df_new

    def _fill_in_missing_column(self, df: pd.DataFrame, column_name: str, value: Scalar) -> None:
        if column_name not in df.columns:
            df[column_name] = value
