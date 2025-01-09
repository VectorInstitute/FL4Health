from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset

from research.rxrx1.data.dataset import Rxrx1Dataset


def label_frequency(dataset: Rxrx1Dataset | Subset) -> None:
    """
    Prints the frequency of each label in the dataset.
    """
    # Extract metadata and label map
    if isinstance(dataset, Rxrx1Dataset):
        metadata, label_map = dataset.metadata, dataset.label_map
    elif isinstance(dataset, Subset):
        assert isinstance(dataset.dataset, Rxrx1Dataset), "Subset dataset must be an Rxrx1Dataset instance."
        metadata, label_map = dataset.dataset.metadata.iloc[list(dataset.indices)], dataset.dataset.label_map
    else:
        raise TypeError("Dataset must be of type Rxrx1Dataset or Subset containing an Rxrx1Dataset.")

    # Count label frequencies
    label_counts = metadata["mapped_label"].value_counts()

    # Print frequency with original labels
    for label, count in label_counts.items():
        original_label = next((k for k, v in label_map.items() if v == label), "Unknown")
        print(f"Label {label} (original: {original_label}): {count} samples")


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

    # Create subsets
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)

    return train_subset, val_subset


def load_rxrx1_data(
    data_path: Path, client_num: int, batch_size: int, seed: int | None = None, train_val_split: float = 0.8
) -> tuple[DataLoader, DataLoader, dict[str, int]]:

    # Read the CSV file
    data = pd.read_csv(f"{data_path}/clients/meta_data_{client_num+1}.csv")

    dataset = Rxrx1Dataset(metadata=data, root=data_path, dataset_type="train", transform=None)

    train_set, validation_set = create_splits(dataset, seed=seed, train_fraction=train_val_split)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_set, batch_size=batch_size)
    num_examples = {
        "train_set": len(train_set),
        "validation_set": len(validation_set),
    }

    return train_loader, validation_loader, num_examples


def load_rxrx1_test_data(data_path: Path, client_num: int, batch_size: int) -> tuple[DataLoader, dict[str, int]]:

    # Read the CSV file
    data = pd.read_csv(f"{data_path}/clients/meta_data_{client_num+1}.csv")

    evaluation_set = Rxrx1Dataset(metadata=data, root=data_path, dataset_type="test", transform=None)

    evaluation_loader = DataLoader(evaluation_set, batch_size=batch_size, shuffle=False)
    num_examples = {"eval_set": len(evaluation_set)}
    return evaluation_loader, num_examples
