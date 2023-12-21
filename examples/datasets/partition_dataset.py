import argparse
import json
import os
import shutil
from pathlib import Path

from examples.datasets.dataset_partitioners import construct_dataset_partitioner


def create_or_overwrite_destination_dir(destination_dir: Path, overwrite: bool) -> None:
    if os.path.exists(destination_dir):
        if overwrite:
            shutil.rmtree(destination_dir)
        else:
            raise ValueError(
                f"The destination folder ({destination_dir}) already exists."
                "If you want to overwrite, use the --overwrite option"
            )
    os.mkdir(destination_dir)


def load_dataset_and_partition(
    dataset_path: Path, partition_dir: Path, partitioner_config_path: Path, n_partitions: int
) -> None:
    with partitioner_config_path.open() as config_path:
        partitioner_config = json.load(config_path)
    dataset_partitioner = construct_dataset_partitioner(dataset_path, partition_dir, partitioner_config)
    dataset_partitioner.partition_dataset(
        n_partitions, "label", {1: "WORLD", 2: "SPORTS", 3: "BUSINESS", 4: "TECHNOLOGY"}
    )


def main(
    original_path_str: str,
    partition_dir_str: str,
    overwrite: bool,
    n_partitions: int,
    partitioner_config_path_str: str,
) -> None:
    original_path = Path(original_path_str)
    assert os.path.exists(original_path)
    partitioner_config_path = Path(partitioner_config_path_str)
    partition_dir = Path(partition_dir_str)
    create_or_overwrite_destination_dir(partition_dir, overwrite)
    load_dataset_and_partition(original_path, partition_dir, partitioner_config_path, n_partitions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Partition Dataset Script")
    parser.add_argument(
        "--original_dataset_path",
        action="store",
        type=str,
        help="Path of the dataset to be partitioned",
        required=True,
    )
    parser.add_argument(
        "--partition_dir",
        action="store",
        type=str,
        help="Path to the destination for the partitions to be created",
        required=True,
    )
    parser.add_argument(
        "--n_partitions", type=int, action="store", required=True, help="Specifies the number of partitions to create"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Specifies that the partition path should be overwritten"
    )
    parser.add_argument(
        "--partitioner_config_path",
        action="store",
        type=str,
        help="Path to the partitioner configuration json",
        required=True,
    )
    args = parser.parse_args()
    main(
        args.original_dataset_path,
        args.partition_dir,
        args.overwrite,
        args.n_partitions,
        args.partitioner_config_path,
    )
