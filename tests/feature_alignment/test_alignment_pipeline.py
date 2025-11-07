import pandas as pd
import pytest

from fl4health.feature_alignment.tab_features_info_encoder import TabularFeaturesInfoEncoder
from fl4health.feature_alignment.tab_features_preprocessor import TabularFeaturesPreprocessor
from fl4health.feature_alignment.tabular_type import TabularType


@pytest.fixture
def create_df1() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ID": [1, 2, 3, 4],
            "Name": [
                "Braund, Mr. Owen Harris",
                "Allen, Mr. William Henry",
                "Bonnell, Miss. Elizabeth",
                "de La Mole, Mathilde",
            ],
            "Age": [22, 35, 58, 20],
            "Sex": ["male", "male", "female", "female"],
            "Wealth": [10.5, 20.4, 30.2, 300],
        }
    )


@pytest.fixture
def create_df2() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ID": [1, 2, 3, 4],
            "Name": [
                "Braund, Mr. Owen Harris",
                "Allen, Mr. William Henry",
                "Bonnell, Miss. Elizabeth",
                "de La Mole, Mathilde",
            ],
            "Age": [22, 35, 58, 20],
            "Sex": ["male", "male", "female", "female"],
            "Insurance": ["Private", "Public", "Private", "NA"],
            "Height": [180.2, 175.6, 174.0, 175.0],
            "Wealth": [10.5, 20.4, 30.2, 300],
        }
    )


@pytest.fixture
def create_df3() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ID": [1, 2, 3, 4],
            "Name": [
                "Braund, Mr. Owen Harris",
                "Allen, Mr. William Henry",
                "Bonnell, Miss. Elizabeth",
                "de La Mole, Mathilde",
            ],
            "Age": [22, 35, 58, 20],
            "Sex": ["male", "male", "female", "female"],
            "Insurance": ["Private", "Public", "Mixed", "NA"],
            "Height": [180.2, 175.6, 174.0, 175.0],
            "Wealth": [10.5, 20.4, 30.2, 300],
        }
    )


def test_encoding(create_df1: pd.DataFrame) -> None:
    df = create_df1
    # Test some basic functionalities of TabularFeaturesInfoEncoder.
    encoder = TabularFeaturesInfoEncoder.encoder_from_dataframe(df, id_column="ID", target_columns="Wealth")
    assert encoder.get_feature_columns() == ["Age", "Name", "Sex"]
    assert encoder.get_target_columns() == ["Wealth"]
    # Note that since the dataframe is very small (number of unique values is small),
    # both "Name" and "Age" were inferred to be of type
    #  TabularType.ORDINAL
    assert len(encoder.features_by_type(TabularType.ORDINAL)) == 2
    assert len(encoder.features_by_type(TabularType.BINARY)) == 1
    assert len(encoder.features_by_type(TabularType.STRING)) == 0
    assert len(encoder.features_by_type(TabularType.NUMERIC)) == 0

    tabular_targets = encoder.get_tabular_targets()
    assert len(tabular_targets) == 1
    assert tabular_targets[0].get_feature_type() == TabularType.NUMERIC
    assert encoder.get_target_dimension() == 1

    # Test that to_json and from_json methods work as expected.
    json_string = encoder.to_json()

    encoder_new = TabularFeaturesInfoEncoder.from_json(json_string)
    assert encoder_new.get_feature_columns() == ["Age", "Name", "Sex"]
    assert encoder_new.get_target_columns() == ["Wealth"]
    assert len(encoder_new.features_by_type(TabularType.ORDINAL)) == 2
    assert len(encoder_new.features_by_type(TabularType.BINARY)) == 1
    assert len(encoder_new.features_by_type(TabularType.STRING)) == 0
    assert len(encoder_new.features_by_type(TabularType.NUMERIC)) == 0

    tabular_targets_new = encoder_new.get_tabular_targets()
    assert len(tabular_targets_new) == 1
    assert tabular_targets_new[0].get_feature_type() == TabularType.NUMERIC
    assert encoder_new.get_target_dimension() == 1


def test_preprocessor(create_df1: pd.DataFrame, create_df2: pd.DataFrame, create_df3: pd.DataFrame) -> None:
    # These three dataframes are misaligned in the following ways:
    #   - df2 contains columns that df1 does not have.
    #   - df3 has the same columns as df2,
    #     but its "Insurance" column, which is a  categorical column,
    #     has more categories than that of df2.
    # These are the major types of misalignment in Tabular data which we care about.
    df1 = create_df1
    df2 = create_df2
    df3 = create_df3

    encoder1 = TabularFeaturesInfoEncoder.encoder_from_dataframe(df1, id_column="ID", target_columns="Wealth")
    encoder2 = TabularFeaturesInfoEncoder.encoder_from_dataframe(df2, id_column="ID", target_columns="Wealth")
    encoder3 = TabularFeaturesInfoEncoder.encoder_from_dataframe(df3, id_column="ID", target_columns="Wealth")

    preprocessor1 = TabularFeaturesPreprocessor(encoder1)
    preprocessor2 = TabularFeaturesPreprocessor(encoder2)
    preprocessor3 = TabularFeaturesPreprocessor(encoder3)

    preprocessors = [preprocessor1, preprocessor2, preprocessor3]

    # Test that features are properly aligned in every type of misalignment.
    for preprocessor in preprocessors:
        features1, targets1 = preprocessor.preprocess_features(df1)
        features2, targets2 = preprocessor.preprocess_features(df2)
        features3, targets3 = preprocessor.preprocess_features(df3)

        assert features1.shape == features2.shape
        assert features2.shape == features3.shape
        assert targets1.shape == targets2.shape
        assert targets2.shape == targets3.shape
