import copy
import os
import pickle
from collections import defaultdict
from collections.abc import Callable
from logging import INFO
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from flwr.common.logger import log
from torch.utils.data import DataLoader, Subset

from fl4health.utils.dataset import TensorDataset


def construct_rxrx1_tensor_dataset(
    metadata: pd.DataFrame,
    data_path: Path,
    client_num: int,
    dataset_type: str,
    transform: Callable | None = None,
) -> tuple[TensorDataset, dict[int, int]]:
    """
    Construct a TensorDataset for rxrx1 data.

    Args:
        metadata (DataFrame): A DataFrame containing image metadata.
        data_path (Path): Root directory which the image data should be loaded.
        client_num (int): Client number to load data for.
        dataset_type (str): 'train' or 'test' to specify dataset type.
        transform (Callable | None): Transformation function to apply to the images. Defaults to None.

    Returns:
        tuple[TensorDataset, dict[int, int]]: A TensorDataset containing the processed images and label map.

    """

    label_map = {label: idx for idx, label in enumerate(sorted(metadata["sirna_id"].unique()))}
    original_label_map = {new_label: original_label for original_label, new_label in label_map.items()}
    metadata = metadata[metadata["dataset"] == dataset_type]
    targets_tensor = torch.Tensor(list(metadata["sirna_id"].map(label_map)))
    data_list = []
    for index in range(len(targets_tensor)):
        with open(
            os.path.join(data_path, f"clients/{dataset_type}_data_{client_num+1}/image_{index}.pkl"), "rb"
        ) as file:
            data_list.append(torch.Tensor(pickle.load(file)).unsqueeze(0))
    data_tensor = torch.cat(data_list)
    return TensorDataset(data_tensor, targets_tensor, transform), original_label_map


def label_frequency(dataset: TensorDataset | Subset, original_label_map: dict[int, int]) -> None:
    """
    Prints the frequency of each label in the dataset.

    Args:
        dataset (TensorDataset | Subset): The dataset to analyze.
        original_label_map (dict[int, int]): A mapping of the original labels to their new labels.

    """
    # Extract metadata and label map
    if isinstance(dataset, TensorDataset):
        targets = dataset.targets
    elif isinstance(dataset, Subset):
        assert isinstance(dataset.dataset, TensorDataset), "Subset dataset must be an TensorDataset instance."
        targets = dataset.dataset.targets
    else:
        raise TypeError("Dataset must be of type TensorDataset or Subset containing an TensorDataset.")

    # Count label frequencies
    label_to_indices = defaultdict(list)
    assert isinstance(targets, torch.Tensor)
    for idx, label in enumerate(targets):  # Assumes dataset[idx] returns (data, label)
        label_to_indices[label].append(idx)

    # Print frequency of labels their names
    for label, count in label_to_indices.items():
        assert isinstance(label, int)
        original_label = original_label_map.get(label)
        log(INFO, f"Label {label} (original: {original_label}): {len(count)} samples")


def create_splits(
    dataset: TensorDataset, seed: int | None = None, train_fraction: float = 0.8
) -> tuple[list[int], list[int]]:
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
    assert isinstance(dataset.targets, torch.Tensor)
    for idx, label in enumerate(dataset.targets):  # Assumes dataset[idx] returns (data, label)
        label_to_indices[label.item()].append(idx)

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

    return train_indices, val_indices


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

    dataset, _ = construct_rxrx1_tensor_dataset(data, data_path, client_num, "train")

    train_indices, val_indices = create_splits(dataset, seed=seed, train_fraction=train_val_split)
    train_set = copy.deepcopy(dataset)
    train_set.data = train_set.data[train_indices]
    assert train_set.targets is not None
    train_set.targets = train_set.targets[train_indices]

    validation_set = copy.deepcopy(dataset)
    validation_set.data = validation_set.data[val_indices]
    assert validation_set.targets is not None
    validation_set.targets = validation_set.targets[val_indices]

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    validation_loader = DataLoader(validation_set, batch_size=batch_size)
    num_examples = {
        "train_set": len(train_set.data),
        "validation_set": len(validation_set.data),
    }

    return train_loader, validation_loader, num_examples


def load_rxrx1_test_data(
    data_path: Path, client_num: int, batch_size: int, num_workers: int = 0
) -> tuple[DataLoader, dict[str, int]]:

    # Read the CSV file
    data = pd.read_csv(f"{data_path}/clients/meta_data_{client_num+1}.csv")

    dataset, _ = construct_rxrx1_tensor_dataset(data, data_path, client_num, "test")

    evaluation_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    num_examples = {"eval_set": len(dataset.data)}
    return evaluation_loader, num_examples
