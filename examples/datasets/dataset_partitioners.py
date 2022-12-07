import os
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Dict, List, cast

import numpy as np
import pandas as pd


class DatasetPartitionerEnum(Enum):
    JSON_TO_PANDAS = "JSON_TO_PANDAS"


class DatasetPartitioner(ABC):
    def __init__(self, dataset_path: Path, partition_dir: Path) -> None:
        super().__init__()
        self.dataset_path = dataset_path
        self.partition_dir = partition_dir

    @abstractmethod
    def partition_dataset(self, n_partiions: int) -> None:
        pass


class JsonToPandasDatasetPartitioner(DatasetPartitioner):
    def __init__(self, dataset_path: Path, partition_dir: Path, config: Dict[str, str]) -> None:
        self._parse_config(config)
        super().__init__(dataset_path, partition_dir)

    def _parse_config(self, config: Dict[str, str]) -> None:
        if "json_lines" in config:
            self.json_lines = config["json_lines"] == "True"

    def partition_dataset(self, n_partitions: int) -> None:
        df = pd.read_json(self.dataset_path, lines=self.json_lines)
        # Shuffle the dataframe rows
        df = df.sample(frac=1).reset_index(drop=True)
        paritioned_dfs = cast(List[pd.DataFrame], np.array_split(df, n_partitions))

        for chunk, df in enumerate(paritioned_dfs):
            df.to_json(
                os.path.join(self.partition_dir, f"partition_{str(chunk)}.json"),
                orient="records",
                lines=self.json_lines,
            )


def construct_dataset_partitioner(
    dataset_path: Path, partition_dir: Path, config: Dict[str, str]
) -> DatasetPartitioner:
    data_loader_enum = DatasetPartitionerEnum(config["dataset_partitioner_type"])
    if data_loader_enum == DatasetPartitionerEnum.JSON_TO_PANDAS:
        return JsonToPandasDatasetPartitioner(dataset_path, partition_dir, config)
    else:
        raise NotImplementedError("No valid partitioner implemented for the configuration")
