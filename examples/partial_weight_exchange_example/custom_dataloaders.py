import math
from typing import List, Tuple, Union

import datasets
import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import Subset
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast


def make_train_val_datasets(dataset: datasets.Dataset, split_ratio: float) -> Tuple[Subset, Subset]:
    assert 0.0 < split_ratio < 1.0
    # cut the train set into train/val using split ratio as the percentage to put into train.
    original_length = len(dataset)
    train_length = math.floor(original_length * split_ratio)
    lengths = [train_length, original_length - train_length]
    result = torch.utils.data.random_split(dataset, lengths)
    return result[0], result[1]


def split_client_data(dataset: datasets.Dataset, num_clients: int, seed: int = 42) -> List[Subset]:
    ratios = np.random.dirichlet(np.ones(num_clients), 1)
    ratios = ratios.reshape(ratios.shape[1])
    generator = torch.Generator().manual_seed(seed)
    return torch.utils.data.random_split(dataset, ratios.tolist(), generator=generator)


def setup_datasets(
    train_split_ratio: float,  # This value is used only the dataset is missing a "validation" split.
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    dataset_name: str,
    num_clients: int,
) -> Tuple[List[Subset], List[Subset], List[Subset]]:
    dataset_dict = load_dataset(dataset_name)
    assert isinstance(dataset_dict, datasets.DatasetDict)

    # Tokenize the text data using the model tokenizer
    tokenized_dataset_dict = dataset_dict.map(
        lambda row: tokenizer(row["text"], truncation=True, padding="max_length"), batched=True
    )

    train_dataset_total = tokenized_dataset_dict["train"]
    # Ensure that the dataloader yields PyTorch tensors, not lists of lists.
    train_dataset_total.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    if "validation" in tokenized_dataset_dict.keys():
        val_dataset_total = tokenized_dataset_dict["validation"]
        val_dataset_total.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    else:
        # Some datasets (e.g., AG news) just has train and test sets (no validation set)
        # split the original training dataset into a training and validation set.
        train_dataset_total, val_dataset_total = make_train_val_datasets(train_dataset_total, train_split_ratio)

    # Create the test set.
    test_dataset_total = tokenized_dataset_dict["test"]
    test_dataset_total.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    train_datasets = split_client_data(train_dataset_total, num_clients)
    val_datasets = split_client_data(val_dataset_total, num_clients)
    test_datasets = split_client_data(test_dataset_total, num_clients)

    return train_datasets, val_datasets, test_datasets
