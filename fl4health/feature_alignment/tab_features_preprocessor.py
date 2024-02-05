from logging import WARNING
from typing import Dict, List, Tuple

import pandas as pd
from flwr.common.logger import log
from flwr.common.typing import NDArray, Scalar
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder

from fl4health.feature_alignment.string_columns_transformer import TextColumnTransformer
from fl4health.feature_alignment.tab_features_info_encoder import TabularFeaturesInfoEncoder
from fl4health.feature_alignment.tabular_feature import MetaData, TabularFeature
from fl4health.feature_alignment.tabular_type import TabularType


class TabularFeaturesPreprocessor:
    """
    TabularFeaturesPreprocessor is responsible for constructing
    the appropriate column transformers based on the information
    encoded in tab_feature_encoder. These transformers will
    then be applied to a pandas dataframe.

    Each tabular feature, which corresponds to a column
    in the pandas dataframe, has its own column transformer. A default
    transformer is initialized for each feature based on its data type,
    but the user may also manually specify a transformer for this
    feature.

    Args:
        tab_feature_encoder (TabularFeaturesInfoEncoder):
        encodes the information necessary for constructing the column transformers.
    """

    def __init__(self, tab_feature_encoder: TabularFeaturesInfoEncoder) -> None:
        self.features_to_pipelines: Dict[str, Pipeline] = {}
        self.targets_to_pipelines: Dict[str, Pipeline] = {}

        self.tabular_features = tab_feature_encoder.get_tabular_features()
        self.tabular_targets = tab_feature_encoder.get_tabular_targets()

        self.feature_columns = tab_feature_encoder.get_feature_columns()
        self.target_columns = tab_feature_encoder.get_target_columns()

        self.features_to_pipelines = self.initialize_default_pipelines(self.tabular_features, one_hot=True)
        self.targets_to_pipelines = self.initialize_default_pipelines(self.tabular_targets, one_hot=False)

        self.data_column_transformer = self.return_column_transformer(self.features_to_pipelines)
        self.target_column_transformer = self.return_column_transformer(self.targets_to_pipelines)

    def get_default_numeric_pipeline(self) -> Pipeline:
        return Pipeline(steps=[("imputer", SimpleImputer(strategy="mean")), ("scaler", MinMaxScaler())])

    def get_default_binary_pipeline(self) -> Pipeline:
        return Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OrdinalEncoder())])

    def get_default_one_hot_pipeline(self, categories: MetaData) -> Pipeline:
        return Pipeline(steps=[("encoder", OneHotEncoder(handle_unknown="ignore", categories=[categories]))])

    def get_default_ordinal_pipeline(self, categories: MetaData) -> Pipeline:
        return Pipeline(
            steps=[
                (
                    "encoder",
                    OrdinalEncoder(
                        unknown_value=len(categories) + 1,
                        handle_unknown="use_encoded_value",
                        categories=[categories],
                    ),
                )
            ]
        )

    def get_default_string_pipeline(self, vocabulary: MetaData) -> Pipeline:
        return Pipeline(steps=[("vectorizer", TextColumnTransformer(TfidfVectorizer(vocabulary=vocabulary)))])

    def initialize_default_pipelines(
        self, tabular_features: List[TabularFeature], one_hot: bool
    ) -> Dict[str, Pipeline]:
        """
        Initialize a default Pipeline for every data column in tabular_features.

        Args:
            tabular_features (List[TabularFeature]): list of tabular
            features in the data columns.
        """
        columns_to_pipelines = {}
        for tab_feature in tabular_features:
            feature_type = tab_feature.get_feature_type()
            feature_name = tab_feature.get_feature_name()
            if feature_type == TabularType.NUMERIC:
                feature_pipeline = self.get_default_numeric_pipeline()
            elif feature_type == TabularType.BINARY:
                feature_pipeline = self.get_default_binary_pipeline()
            elif feature_type == TabularType.ORDINAL:
                feature_categories = tab_feature.get_metadata()
                if one_hot:
                    feature_pipeline = self.get_default_one_hot_pipeline(feature_categories)
                else:
                    feature_pipeline = self.get_default_ordinal_pipeline(feature_categories)
            else:
                vocabulary = tab_feature.get_metadata()
                feature_pipeline = self.get_default_string_pipeline(vocabulary)
            columns_to_pipelines[feature_name] = feature_pipeline
        return columns_to_pipelines

    def return_column_transformer(self, pipelines: Dict[str, Pipeline]) -> ColumnTransformer:
        transformers = [
            (f"{feature_name}_pipeline", pipelines[feature_name], [feature_name])
            for feature_name in sorted(pipelines.keys())
        ]
        # If a column does not have an associated transformer then it is dropped from the df.
        return ColumnTransformer(
            transformers=transformers,
            remainder="drop",
        )

    def set_feature_pipeline(self, feature_name: str, pipeline: Pipeline) -> None:
        # This method allows the user to customize a specific pipeline to be applied to a specific feature.
        # For example, the user may want to use different scalers for two distinct numerical features.
        if feature_name in self.features_to_pipelines:
            self.features_to_pipelines[feature_name] = pipeline
            self.data_column_transformer = self.return_column_transformer(self.features_to_pipelines)
        elif feature_name in self.targets_to_pipelines:
            self.targets_to_pipelines[feature_name] = pipeline
            self.target_column_transformer = self.return_column_transformer(self.targets_to_pipelines)
        else:
            log(WARNING, f"{feature_name} is neither a feature nor target and the provided pipeline will be ignored.")

    def preprocess_features(self, df: pd.DataFrame) -> Tuple[NDArray, NDArray]:
        # If the dataframe has an entire column missing, we need to fill it with some default value first.
        df_filled = self.fill_in_missing_columns(df)
        # After filling in missing columns, apply the feature alignment transform.
        return self.data_column_transformer.fit_transform(
            df_filled[self.feature_columns]
        ), self.target_column_transformer.fit_transform(df_filled[self.target_columns])

    def fill_in_missing_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return a new DataFrame where entire missing columns
        are filled with values specified in each column's default fill value.
        """
        df_new = df.copy(deep=True)
        for tab_feature in self.tabular_features:
            self._fill_in_missing_column(df_new, tab_feature.get_feature_name(), tab_feature.get_fill_value())
        return df_new

    def _fill_in_missing_column(self, df: pd.DataFrame, column_name: str, value: Scalar) -> None:
        if column_name not in df.columns:
            df[column_name] = value
