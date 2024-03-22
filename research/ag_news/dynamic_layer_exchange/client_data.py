from functools import partial
from typing import Dict, List, Tuple

import datasets
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import BertTokenizer, PreTrainedTokenizer

from fl4health.utils.dataset import TextClassificationDataset
from fl4health.utils.sampler import DirichletLabelBasedSampler


def collate_fn_with_padding(
    tokenizer: PreTrainedTokenizer, batch: List[Tuple[Dict[str, List[torch.Tensor]], torch.Tensor]]
) -> Tuple[Dict[str, List[torch.Tensor]], torch.Tensor]:
    input_list, target_list = [], []
    for pair in batch:
        input_list.append(pair[0])
        target_list.append(pair[1])
    return tokenizer.pad(input_list), torch.stack(target_list)


def create_dict_dataset(
    dataset: datasets.Dataset, column_names: List[str], target_name: str
) -> TextClassificationDataset:
    data_dict = {}
    for column_name in column_names:
        data_dict[column_name] = dataset[column_name]
    targets = dataset[target_name]
    return TextClassificationDataset(data=data_dict, targets=targets)


def construct_dataloaders(batch_size: int, sample_percentage: float, beta: float) -> Tuple[DataLoader, DataLoader]:

    assert 0 <= sample_percentage <= 1 and beta > 0
    sampler = DirichletLabelBasedSampler(list(range(4)), sample_percentage=sample_percentage, beta=beta)

    ag_news_dataset = load_dataset("ag_news")
    bert_tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-cased")
    # Apply Bert's tokenizer to all data points. Also perform truncation.
    tokenized_ag_news = ag_news_dataset.map(
        function=lambda x: bert_tokenizer(x["text"], truncation=True),
    )

    # Set the columns of datasets so different parts can be easily accessed.
    train_dataset = tokenized_ag_news["train"]
    column_names = ["input_ids", "attention_mask", "label"]
    train_dataset.set_format(type="torch", columns=column_names)

    # It is important to ensure that the column names do not contain the column
    # that correspond to the raw text because Hugging face does that by default.
    train_dataset = create_dict_dataset(train_dataset, column_names, "label")
    train_dataset = sampler.subsample(train_dataset)

    val_dataset = tokenized_ag_news["test"]
    val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    val_dataset = create_dict_dataset(val_dataset, column_names, "label")
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
