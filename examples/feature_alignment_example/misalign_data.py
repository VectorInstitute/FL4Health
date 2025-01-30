import random
from logging import INFO
from pathlib import Path

import pandas as pd
from flwr.common.logger import log


def random_split_data(df: pd.DataFrame, n: int) -> list[pd.DataFrame]:
    df_rand = df.sample(frac=1.0, random_state=42)
    num_rows_per_df = len(df_rand) // n
    smaller_dfs = [df_rand.iloc[i * num_rows_per_df : (i + 1) * num_rows_per_df] for i in range(n - 1)]
    smaller_dfs.append(df_rand.iloc[(n - 1) * num_rows_per_df :])
    return smaller_dfs


if __name__ == "__main__":
    data_path = Path("examples/datasets/mimic3d/mimic3d.csv")
    target_datapath = "examples/datasets/mimic3d"

    df = pd.read_csv(data_path)
    df.dropna(inplace=True, how="any")
    dfs = random_split_data(df, 2)
    df1 = dfs[0].reset_index()
    df2 = dfs[1].reset_index()

    # Dropping columns to create misalignment.
    columns_to_drop = ["ExpiredHospital", "admit_type", "NumRx", "ethnicity"]
    df2 = df2.drop(columns=columns_to_drop)
    log(INFO, "Hospital2 missing columns: {', '.join(columns_to_drop)}")

    # Now we randomly select 10 percent of the rows of df2
    # and set its 'insurance' column to 'Unknown'
    num_rows_to_select = int(0.10 * len(df2))
    random_indices = random.sample(range(len(df2)), num_rows_to_select)

    # Set a specific column ('insurance' in this example) to the specific value for the selected rows
    df2.loc[random_indices, "insurance"] = "Unknown"

    log(INFO, f"Hospital1 insurance values: {df1['insurance'].unique()}")
    log(INFO, f"Hospital2 insurance values: {df2['insurance'].unique()}")

    df1.to_csv(f"{target_datapath}/mimic3d_hospital1.csv", index=False)
    df2.to_csv(f"{target_datapath}/mimic3d_hospital2.csv", index=False)
