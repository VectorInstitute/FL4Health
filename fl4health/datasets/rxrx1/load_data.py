from collections import defaultdict
from logging import INFO
from pathlib import Path

import numpy as np
import pandas as pd
from flwr.common.logger import log
from torch.utils.data import DataLoader, Subset

from fl4health.datasets.rxrx1.dataset import Rxrx1Dataset


def label_frequency(dataset: Rxrx1Dataset | Subset) -> None:
    """
    Prints the frequency of each label in the dataset.
    """
    # Extract metadata and label map
    if isinstance(dataset, Rxrx1Dataset):
        metadata, original_label_map = dataset.metadata, dataset.original_label_map
    elif isinstance(dataset, Subset):
        assert isinstance(dataset.dataset, Rxrx1Dataset), "Subset dataset must be an Rxrx1Dataset instance."
        metadata, original_label_map = (
            dataset.dataset.metadata.iloc[list(dataset.indices)],
            dataset.dataset.original_label_map,
        )
    else:
        raise TypeError("Dataset must be of type Rxrx1Dataset or Subset containing an Rxrx1Dataset.")

    # Count label frequencies
    label_counts = metadata["mapped_label"].value_counts()

    # Print frequency of labels their names
    for label, count in label_counts.items():
        assert isinstance(label, int)
        original_label = original_label_map.get(label)
        log(INFO, f"Label {label} (original: {original_label}): {count} samples")


def create_splits(
    dataset: Rxrx1Dataset, seed: int | None = None, train_fraction: float = 0.8
) -> tuple[Subset, Subset]:
    """
    Splits the dataset into training and validation sets.

    Args:
        dataset (Dataset): The dataset to split.
        train_fraction (float): Fraction of data to use for training.

    Returns:
        Tuple: (train_dataset, val_dataset)
    """

    # Group indices by label
    label_to_indices = defaultdict(list)
    for idx, label in enumerate(dataset.metadata["mapped_label"]):  # Assumes dataset[idx] returns (data, label)
        label_to_indices[label].append(idx)

    # Stratified splitting
    train_indices, val_indices = [], []
    for label, indices in label_to_indices.items():
        if seed is not None:
            np_generator = np.random.default_rng(seed)
            np_generator.shuffle(indices)
        else:
            np.random.shuffle(indices)
        split_point = int(len(indices) * train_fraction)
        train_indices.extend(indices[:split_point])
        val_indices.extend(indices[split_point:])
        if len(val_indices) == 0:
            log(INFO, "Warning: Validation set is empty. Consider changing the train_fraction parameter.")

    # Create subsets
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)

    return train_subset, val_subset


def load_rxrx1_data(
    data_path: Path,
    client_num: int,
    batch_size: int,
    seed: int | None = None,
    train_val_split: float = 0.8,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, dict[str, int]]:

    # Read the CSV file
    data = pd.read_csv(f"{data_path}/clients/meta_data_{client_num+1}.csv")

    dataset = Rxrx1Dataset(metadata=data, root=data_path, dataset_type="train", transform=None)

    train_set, validation_set = create_splits(dataset, seed=seed, train_fraction=train_val_split)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    validation_loader = DataLoader(validation_set, batch_size=batch_size)
    num_examples = {
        "train_set": len(train_set),
        "validation_set": len(validation_set),
    }

    return train_loader, validation_loader, num_examples


def load_rxrx1_test_data(
    data_path: Path, client_num: int, batch_size: int, num_workers: int = 0
) -> tuple[DataLoader, dict[str, int]]:

    # Read the CSV file
    data = pd.read_csv(f"{data_path}/clients/meta_data_{client_num+1}.csv")

    evaluation_set = Rxrx1Dataset(metadata=data, root=data_path, dataset_type="test", transform=None)

    evaluation_loader = DataLoader(
        evaluation_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    num_examples = {"eval_set": len(evaluation_set)}
    return evaluation_loader, num_examples
