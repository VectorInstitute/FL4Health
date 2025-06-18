import copy

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from fl4health.feature_alignment.string_columns_transformer import TextColumnTransformer, TextMulticolumnTransformer


def create_df_multi() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Name": [
                "Braund, Mr. Owen Harris",
                "Allen, Mr. William Henry",
                "Bonnell, Miss. Elizabeth",
                "de La Mole, Mathilde",
            ],
            "Sex": ["male", "male", "female", "female"],
        }
    )


def create_df_single() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Name": [
                "Braund, Mr. Owen Harris",
                "Allen, Mr. William Henry",
                "Bonnell, Miss. Elizabeth",
                "de La Mole, Mathilde",
            ],
        }
    )


def test_text_multicolumn_transformer() -> None:
    df = create_df_multi()
    df_to_transform = copy.deepcopy(df)
    multicolumn_transformer = TextMulticolumnTransformer(CountVectorizer())
    multicolumn_transformer.fit(df)

    vocabulary = multicolumn_transformer.transformer.vocabulary_

    transformed_df = multicolumn_transformer.transform(df_to_transform)
    vectorized_text = transformed_df.toarray()  # type: ignore

    vectorized_text_row_0 = vectorized_text[0]
    vectorized_text_row_2 = vectorized_text[2]
    assert vectorized_text_row_0[vocabulary["braund"]] == 1
    assert vectorized_text_row_0[vocabulary["allen"]] == 0
    assert vectorized_text_row_0[vocabulary["harris"]] == 1

    assert vectorized_text_row_2[vocabulary["bonnell"]] == 1
    assert vectorized_text_row_2[vocabulary["allen"]] == 0
    assert vectorized_text_row_2[vocabulary["female"]] == 1


def test_text_column_transformer() -> None:
    df = create_df_single()
    df_to_transform = copy.deepcopy(df)
    multicolumn_transformer = TextColumnTransformer(CountVectorizer())
    multicolumn_transformer.fit(df)

    vocabulary = multicolumn_transformer.transformer.vocabulary_

    transformed_df = multicolumn_transformer.transform(df_to_transform)
    vectorized_text = transformed_df.toarray()  # type: ignore

    vectorized_text_row_0 = vectorized_text[0]
    vectorized_text_row_2 = vectorized_text[2]
    assert vectorized_text_row_0[vocabulary["braund"]] == 1
    assert vectorized_text_row_0[vocabulary["allen"]] == 0
    assert vectorized_text_row_0[vocabulary["harris"]] == 1

    assert vectorized_text_row_2[vocabulary["bonnell"]] == 1
    assert vectorized_text_row_2[vocabulary["allen"]] == 0
    assert vectorized_text_row_2[vocabulary["miss"]] == 1
