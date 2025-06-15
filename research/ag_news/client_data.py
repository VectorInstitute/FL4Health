from functools import partial
from pathlib import Path

import datasets
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import BertTokenizer, PreTrainedTokenizer

from fl4health.utils.dataset import DictionaryDataset
from fl4health.utils.sampler import DirichletLabelBasedSampler


def collate_fn_with_padding(
    tokenizer: PreTrainedTokenizer, batch: list[tuple[dict[str, list[torch.Tensor]], torch.Tensor]]
) -> tuple[dict[str, list[torch.Tensor]], torch.Tensor]:
    """Pad the sequences within a batch to the same length."""
    input_list, target_list = [], []
    for input_dict, target in batch:
        input_list.append(input_dict)
        target_list.append(target)
    return tokenizer.pad(input_list).data, torch.stack(target_list)  # type: ignore


def create_text_classification_dataset(
    dataset: datasets.Dataset, column_names: list[str], target_name: str
) -> DictionaryDataset:
    data_dict = {}
    for column_name in column_names:
        data_dict[column_name] = dataset[column_name]
    targets = dataset[target_name]
    return DictionaryDataset(data=data_dict, targets=targets)


def construct_dataloaders(
    batch_size: int, sample_percentage: float, beta: float, data_path: Path | None = None
) -> tuple[DataLoader, DataLoader]:
    assert 0 <= sample_percentage <= 1 and beta > 0
    sampler = DirichletLabelBasedSampler(list(range(4)), sample_percentage=sample_percentage, beta=beta)

    if data_path is None:
        ag_news_dataset = load_dataset("ag_news")
    else:
        ag_news_dataset = load_dataset(name="ag_news", data_dir=data_path)
    bert_tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-cased")
    # Apply Bert's tokenizer to all data points. Also perform truncation.
    tokenized_ag_news = ag_news_dataset.map(
        function=lambda x: bert_tokenizer(x["text"], truncation=True),
    )

    # Set the columns of datasets so different parts can be easily accessed.
    train_dataset = tokenized_ag_news["train"]

    # It is important to ensure that the column names do not contain the column
    # that correspond to the raw text to ensure that padding via collate functions works as expected.
    data_column_names = ["input_ids", "attention_mask"]
    all_column_names = data_column_names + ["label"]
    train_dataset.set_format(type="torch", columns=all_column_names)

    train_dataset = create_text_classification_dataset(train_dataset, data_column_names, "label")
    train_dataset = sampler.subsample(train_dataset)

    val_dataset = tokenized_ag_news["test"]
    val_dataset.set_format(type="torch", columns=all_column_names)
    val_dataset = create_text_classification_dataset(val_dataset, data_column_names, "label")
    val_dataset = sampler.subsample(val_dataset)

    # The collate function is used to dynamically pad the sequences within every batch to the same length.
    # This is more efficient than padding all sequences to the maximum length.
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=partial(collate_fn_with_padding, bert_tokenizer)
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, collate_fn=partial(collate_fn_with_padding, bert_tokenizer)
    )
    return train_loader, val_loader
