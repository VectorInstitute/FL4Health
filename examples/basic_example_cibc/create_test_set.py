from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

ORIGINAL_FILENAME = "fraud_full.csv"
DEFAULT_DIR = Path(__file__).parent.absolute()


def create_test_set():
    df = pd.read_csv(DEFAULT_DIR / ORIGINAL_FILENAME, parse_dates=["timestamp"])
    df.columns = df.columns.str.lower()

    # get rid of binary columns into integer columns
    df = df.replace({True: 1, False: 0})

    train_df, test_df = train_test_split(df, test_size=0.2)
    train_df = cast(pd.DataFrame, train_df)
    test_df = cast(pd.DataFrame, test_df)

    # Identify features to scale (excluding timestamp and target)
    features = [col for col in df.columns if col not in ["timestamp", "fraud_label"]]

    # Fit scaler only on training data
    scaler = StandardScaler()
    scaler.fit(train_df[features])

    # Transform both train and test data
    train_df_scaled = train_df.copy()
    test_df_scaled = test_df.copy()

    train_df_scaled[features] = scaler.transform(train_df[features])
    test_df_scaled[features] = scaler.transform(test_df[features])

    # write to disk
    train_df_scaled.to_csv(DEFAULT_DIR / "fraud_train.csv", index=False)
    test_df_scaled.to_csv(DEFAULT_DIR / "fraud_test.csv", index=False)


if __name__ == "__main__":
    create_test_set()
