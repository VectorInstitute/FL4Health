import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from fl4health.feature_alignment.constants import TextFeatureTransformer


class TextMulticolumnTransformer(BaseEstimator, TransformerMixin):
    """
    The purpose of this class is to enable the application of text feature transformers from sklearn
    to multiple string columns, which is not supported in the first place.
    """

    def __init__(self, transformer: TextFeatureTransformer):
        self.transformer = transformer

    def fit(self, X: pd.DataFrame, y: pd.DataFrame = None) -> "TextMulticolumnTransformer":
        joined_X = X.apply(lambda x: " ".join(x), axis=1)
        self.transformer.fit(joined_X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        joined_X = X.apply(lambda x: " ".join(x), axis=1)
        return self.transformer.transform(joined_X)


class TextColumnTransformer(BaseEstimator, TransformerMixin):
    """
    The purpose of this class is to enable the application of text feature transformers from sklearn
    to a single-column pandas dataframe, which is not supported in the first place.
    """

    def __init__(self, transformer: TextFeatureTransformer):
        self.transformer = transformer

    def fit(self, X: pd.DataFrame, y: pd.DataFrame = None) -> "TextColumnTransformer":
        assert isinstance(X, pd.DataFrame) and X.shape[1] == 1
        self.transformer.fit(X[X.columns[0]])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        assert isinstance(X, pd.DataFrame) and X.shape[1] == 1
        return self.transformer.transform(X[X.columns[0]])
