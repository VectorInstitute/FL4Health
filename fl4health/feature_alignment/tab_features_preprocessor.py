from logging import WARNING

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
    def __init__(self, tab_feature_encoder: TabularFeaturesInfoEncoder) -> None:
        """
        ``TabularFeaturesPreprocessor`` is responsible for constructing the appropriate column transformers based on
        the information encoded in ``tab_feature_encoder``. These transformers will then be applied to a pandas
        dataframe.

        Each tabular feature, which corresponds to a column in the pandas dataframe, has its own column transformer.
        A default transformer is initialized for each feature based on its data type, but the user may also manually
        specify a transformer for this feature.

        Args:
            tab_feature_encoder (TabularFeaturesInfoEncoder): Encodes the information necessary for constructing the
                column transformers.
        """
        self.features_to_pipelines: dict[str, Pipeline] = {}
        self.targets_to_pipelines: dict[str, Pipeline] = {}

        self.tabular_features = tab_feature_encoder.get_tabular_features()
        self.tabular_targets = tab_feature_encoder.get_tabular_targets()

        self.feature_columns = tab_feature_encoder.get_feature_columns()
        self.target_columns = tab_feature_encoder.get_target_columns()

        self.features_to_pipelines = self.initialize_default_pipelines(self.tabular_features, one_hot=True)
        self.targets_to_pipelines = self.initialize_default_pipelines(self.tabular_targets, one_hot=False)

        self.data_column_transformer = self.return_column_transformer(self.features_to_pipelines)
        self.target_column_transformer = self.return_column_transformer(self.targets_to_pipelines)

    def get_default_numeric_pipeline(self) -> Pipeline:
        """
        Default numeric pipeline factory. Mean imputation and default min-max scaler.

        Returns:
            (Pipeline): Default numeric pipeline
        """
        return Pipeline(steps=[("imputer", SimpleImputer(strategy="mean")), ("scaler", MinMaxScaler())])

    def get_default_binary_pipeline(self) -> Pipeline:
        """
        Default binary pipeline factor. Most frequent imputer and an ordinal encoder.

        Returns:
            (Pipeline): Default binary pipeline
        """
        return Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OrdinalEncoder())])

    def get_default_one_hot_pipeline(self, categories: MetaData) -> Pipeline:
        """
        Default one hot encoding pipeline. Unknowns are ignored, categories are provided as an input.

        Args:
            categories (MetaData): Categories to be one hot encoded.

        Returns:
            (Pipeline): Default one-hot encoding pipeline
        """
        return Pipeline(steps=[("encoder", OneHotEncoder(handle_unknown="ignore", categories=[categories]))])

    def get_default_ordinal_pipeline(self, categories: MetaData) -> Pipeline:
        """
        Default ordinal pipeline. Unknowns have a category. Other categories are provided.

        Args:
            categories (MetaData): Categories to be used in encoding

        Returns:
            (Pipeline): Default ordinal pipeline
        """
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
        """
        Default string/text encoding pipeline. The vocabulary is provided and this is used to instantiate a default
        ``TfidfVectorizer``.

        Args:
            vocabulary (MetaData): Vocabulary to serve as the ``TfidfVectorizer`` vocab.

        Returns:
            (Pipeline): Default string/text encoding pipeline.
        """
        return Pipeline(steps=[("vectorizer", TextColumnTransformer(TfidfVectorizer(vocabulary=vocabulary)))])

    def initialize_default_pipelines(
        self, tabular_features: list[TabularFeature], one_hot: bool
    ) -> dict[str, Pipeline]:
        """
        Initialize a default Pipeline for every data column in ``tabular_features``.

        Args:
            tabular_features (list[TabularFeature]): list of tabular features in the data columns.
            one_hot (bool): Whether or not to apply a default one-hot pipeline.

        Returns:
            (dict[str, Pipeline]): Default feature processing pipeline per feature in the list.
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

    def return_column_transformer(self, pipelines: dict[str, Pipeline]) -> ColumnTransformer:
        """
        Given a set of pipelines create a set of column transformations based on those pipelines.

        Args:
            pipelines (dict[str, Pipeline]): Dictionary of pipelines for columns with the keys of the dictionary
                corresponding to the column names

        Returns:
            (ColumnTransformer): Transformer for the specified columns. The unspecified columns are dropped.
        """
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
        """
        This method allows the user to customize a specific pipeline to be applied to a specific feature.
        For example, the user may want to use different scalers for two distinct numerical features.

        Args:
            feature_name (str): target column name in the dataframe to apply the pipeline to
            pipeline (Pipeline): Pipeline to apply to the associated column.
        """
        if feature_name in self.features_to_pipelines:
            self.features_to_pipelines[feature_name] = pipeline
            self.data_column_transformer = self.return_column_transformer(self.features_to_pipelines)
        elif feature_name in self.targets_to_pipelines:
            self.targets_to_pipelines[feature_name] = pipeline
            self.target_column_transformer = self.return_column_transformer(self.targets_to_pipelines)
        else:
            log(WARNING, f"{feature_name} is neither a feature nor target and the provided pipeline will be ignored.")

    def preprocess_features(self, df: pd.DataFrame) -> tuple[NDArray, NDArray]:
        """
        Preprocess the provided dataframe with the specified pipelines.

        Args:
            df (pd.DataFrame): Dataframe to be processed.

        Returns:
            (tuple[NDArray, NDArray]): Resulting input and target numpy arrays after preprocessing.
        """
        # If the dataframe has an entire column missing, we need to fill it with some default value first.
        df_filled = self.fill_in_missing_columns(df)
        # After filling in missing columns, apply the feature alignment transform.
        return (
            self.data_column_transformer.fit_transform(df_filled[self.feature_columns]),
            self.target_column_transformer.fit_transform(df_filled[self.target_columns]),
        )

    def fill_in_missing_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return a new DataFrame where entire missing columns are filled with values specified in each column's
        default fill value.

        Args:
            df (pd.DataFrame): Dataframe to be filled

        Returns:
            (pd.DataFrame): Filled dataframe
        """
        df_new = df.copy(deep=True)
        for tab_feature in self.tabular_features:
            self._fill_in_missing_column(df_new, tab_feature.get_feature_name(), tab_feature.get_fill_value())
        return df_new

    def _fill_in_missing_column(self, df: pd.DataFrame, column_name: str, value: Scalar) -> None:
        if column_name not in df.columns:
            df[column_name] = value
