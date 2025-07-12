import os
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd


class DatasetPartitionerEnum(Enum):
    JSON_TO_PANDAS = "JSON_TO_PANDAS"
    CSV_TO_PANDAS = "CSV_TO_PANDAS"


class DatasetPartitioner(ABC):
    def __init__(self, dataset_path: Path, partition_dir: Path) -> None:
        super().__init__()
        self.dataset_path = dataset_path
        self.partition_dir = partition_dir

    @abstractmethod
    def partition_dataset(
        self, n_partitions: int, label_column_name: str | None = None, label_map: dict[int, str] | None = None
    ) -> None:
        pass


class JsonToPandasDatasetPartitioner(DatasetPartitioner):
    def __init__(self, dataset_path: Path, partition_dir: Path, config: dict[str, str]) -> None:
        self._parse_config(config)
        super().__init__(dataset_path, partition_dir)

    def _parse_config(self, config: dict[str, str]) -> None:
        if "json_lines" in config:
            self.json_lines = config["json_lines"] == "True"

    def partition_dataset(
        self, n_partitions: int, label_column_name: str | None = None, label_map: dict[int, str] | None = None
    ) -> None:
        df = pd.read_json(self.dataset_path, lines=self.json_lines)
        # Shuffle the dataframe rows
        df = df.sample(frac=1).reset_index(drop=True)
        partitioned_dfs = cast(list[pd.DataFrame], np.array_split(df, n_partitions))

        for chunk, df in enumerate(partitioned_dfs):
            df.to_json(
                os.path.join(self.partition_dir, f"partition_{str(chunk)}.json"),
                orient="records",
                lines=self.json_lines,
            )


class CsvToPandasDatasetPartitioner(DatasetPartitioner):
    def partition_dataset(
        self, n_partitions: int, label_column_name: str | None = None, label_map: dict[int, str] | None = None
    ) -> None:
        df = pd.read_csv(self.dataset_path, names=["label", "title", "body"])
        # Shuffle the dataframe rows
        df = df.sample(frac=1).reset_index(drop=True)
        if label_column_name and label_map:
            df["category"] = df[label_column_name].map(label_map)
        partitioned_dfs = cast(list[pd.DataFrame], np.array_split(df, n_partitions))

        for chunk, df in enumerate(partitioned_dfs):
            df.to_json(
                os.path.join(self.partition_dir, f"partition_{str(chunk)}.json"),
                orient="records",
                lines=True,
            )


def construct_dataset_partitioner(
    dataset_path: Path, partition_dir: Path, config: dict[str, str]
) -> DatasetPartitioner:
    data_loader_enum = DatasetPartitionerEnum(config["dataset_partitioner_type"])
    if data_loader_enum == DatasetPartitionerEnum.JSON_TO_PANDAS:
        return JsonToPandasDatasetPartitioner(dataset_path, partition_dir, config)
    if data_loader_enum == DatasetPartitionerEnum.CSV_TO_PANDAS:
        return CsvToPandasDatasetPartitioner(dataset_path, partition_dir)
    raise NotImplementedError("No valid partitioner implemented for the configuration")
