from __future__ import annotations

import json
from pathlib import Path

import nltk
import numpy as np
import pandas as pd
import torch
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


nltk.download("punkt_tab")


class LabelEncoder:
    def __init__(self, classes: list[str], label_to_class: dict[int, str], class_to_label: dict[str, int]) -> None:
        self.classes = classes
        self.label_to_class = label_to_class
        self.class_to_label = class_to_label

    @staticmethod
    def encoder_from_dataframe(df: pd.DataFrame, class_column: str) -> LabelEncoder:
        categories = df[class_column].astype("category")
        categories_str = [str(category) for category in categories.to_list()]
        label_to_class = dict(set(zip(categories.cat.codes, categories_str)))
        class_to_label = {category: label for label, category in label_to_class.items()}
        classes = categories.unique().tolist()
        return LabelEncoder(classes, label_to_class, class_to_label)

    @staticmethod
    def from_json(json_str: str) -> LabelEncoder:
        attributes = json.loads(json_str)
        # need to cast string keys to int
        label_to_class = {int(label): category for label, category in json.loads(attributes["label_to_class"]).items()}
        return LabelEncoder(
            json.loads(attributes["classes"]),
            label_to_class,
            {category: label for label, category in label_to_class.items()},
        )

    def to_json(self) -> str:
        return json.dumps(
            {
                "label_to_class": json.dumps(self.label_to_class),
                "classes": json.dumps(self.classes),
            }
        )

    def label_dataframe(self, df: pd.DataFrame, class_column: str) -> pd.DataFrame:
        df["label"] = df[class_column].map(self.class_to_label)
        return df


class Vocabulary:
    def __init__(self, vocabulary_dict: dict[str, int] | None, train_set: list[list[str]] | None) -> None:
        if vocabulary_dict is not None:
            self.word2index = vocabulary_dict
        elif train_set is not None:
            self._create_vocabulary(train_set)
        else:
            raise ValueError("Must provide either precomputed dictionary or training set to create vocabulary")
        self.vocabulary_size = len(self.word2index)

    def _create_vocabulary(self, train_set: list[list[str]]) -> None:
        word2index = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        current_index = 4
        for tokenized_text in train_set:
            for token in tokenized_text:
                if token not in word2index:
                    word2index[token] = current_index
                    current_index += 1
        self.word2index = word2index

    def encode_and_pad(self, tokenized_text: list[str], seq_length: int) -> list[int]:
        sos = [self.word2index["<SOS>"]]
        eos = [self.word2index["<EOS>"]]
        pad = [self.word2index["<PAD>"]]

        if len(tokenized_text) < seq_length - 2:  # -2 for SOS and EOS
            n_pads = seq_length - 2 - len(tokenized_text)
            encoded = [self.get_index_or_unk(w) for w in tokenized_text]
            return sos + encoded + eos + pad * n_pads
        # truncating if needed
        encoded = [self.get_index_or_unk(w) for w in tokenized_text]
        truncated = encoded[: seq_length - 2]
        return sos + truncated + eos

    def get_index_or_unk(self, token: str) -> int:
        if token in self.word2index:
            return self.word2index[token]
        return self.word2index["<UNK>"]

    def to_json(self) -> str:
        return json.dumps(self.word2index)

    @staticmethod
    def from_json(json_str: str) -> Vocabulary:
        return Vocabulary(json.loads(json_str), None)


def tokenize_labeled_text(df: pd.DataFrame) -> list[tuple[int, list[str]]]:
    # Assumes the dataframe has two columns (label and text to be tokenized)
    return [(label, word_tokenize(text)) for label, text in list(df.to_records(index=False))]


def get_local_data(path: Path) -> pd.DataFrame:
    return pd.read_json(path, lines=True)


def create_weight_matrix(train_df: pd.DataFrame) -> torch.Tensor:
    value_counts = train_df["label"].value_counts().sort_index()
    maximum_value = torch.ones(len(value_counts)) * value_counts.max()
    return maximum_value.div(torch.tensor(value_counts.values))


def construct_dataloaders(
    path: Path, vocabulary: Vocabulary, label_encoder: LabelEncoder, sequence_length: int, batch_size: int
) -> tuple[DataLoader, DataLoader, dict[str, int], torch.Tensor]:
    df = get_local_data(path)
    # lower case the headlines and description and concatenate
    df["article_text"] = df["title"].str.lower() + " " + df["body"].str.lower()
    # encode category column as labels
    df = label_encoder.label_dataframe(df, "category")
    processed_df = df[["label", "article_text"]]

    train_df, validation_df = train_test_split(processed_df, test_size=0.2)

    train_set = tokenize_labeled_text(train_df)
    val_set = tokenize_labeled_text(validation_df)
    weight_matrix = create_weight_matrix(train_df)

    train_encoded = [
        (vocabulary.encode_and_pad(tokenized_text, sequence_length), label) for label, tokenized_text in train_set
    ]
    val_encoded = [
        (vocabulary.encode_and_pad(tokenized_text, sequence_length), label) for label, tokenized_text in val_set
    ]

    train_x = np.array([encoded_text for encoded_text, _ in train_encoded])
    train_y = np.array([label for _, label in train_encoded])
    val_x = np.array([encoded_text for encoded_text, _ in val_encoded])
    val_y = np.array([label for _, label in val_encoded])

    train_ds = TensorDataset(torch.from_numpy(train_x), torch.tensor(train_y, dtype=torch.long))
    val_ds = TensorDataset(torch.from_numpy(val_x), torch.tensor(val_y, dtype=torch.long))

    train_dl = DataLoader(train_ds, shuffle=True, batch_size=batch_size, drop_last=True)
    val_dl = DataLoader(val_ds, shuffle=True, batch_size=batch_size, drop_last=True)

    num_examples = {
        "train_set": len(train_dl),
        "validation_set": len(val_dl),
    }

    return train_dl, val_dl, num_examples, weight_matrix
