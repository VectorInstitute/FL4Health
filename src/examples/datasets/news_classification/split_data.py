import os
from typing import List, cast

import numpy as np
import pandas as pd

ORIGINAL_DATASET_PATH = os.path.join(os.path.dirname(__file__), "news_dataset.json")
DISTRIBUTED_DATASET_DIR = os.path.join(os.path.dirname(__file__), "distributed_datasets")

news_df = pd.read_json(ORIGINAL_DATASET_PATH, lines=True)
news_df.drop(["authors", "link"], axis=1, inplace=True)
news_df["date"] = pd.to_datetime(news_df["date"])
# lower case the headlines
news_df["article_text"] = news_df["headline"].str.lower() + " " + news_df["short_description"].str.lower()
news_df = news_df.sample(frac=1).reset_index(drop=True)
distributed_news_dfs = cast(List[pd.DataFrame], np.array_split(news_df, 3))

for chunk, df in enumerate(distributed_news_dfs):
    df.to_json(os.path.join(DISTRIBUTED_DATASET_DIR, f"partition_{str(chunk)}.json"), orient="records", lines=True)
