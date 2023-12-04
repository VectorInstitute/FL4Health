from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import torch
import torchtext.transforms as T
from torch.utils.data import DataLoader, Dataset, IterableDataset, random_split
from torchdata.datapipes.iter import IterableWrapper
from torchtext.datasets import AG_NEWS
from torchtext.functional import to_tensor
from torchtext.models import ROBERTA_BASE_ENCODER

from fl4health.utils.dataset import BaseDataset
from fl4health.utils.sampler import DirichletLabelBasedSampler

PAD = 1


class ListDataset(Dataset):
    def __init__(self, data: List[Any]) -> None:
        self.data = data

    def __getitem__(self, index: int) -> Any:
        return self.data[index]

    def __len__(self) -> int:
        return len(self.data)


class TextClassificationDataset(BaseDataset):
    """
    The Purpose of this class is twofold:
    1. Unify the length of all input sequences.
    2. Reformat the dataset so we can perform Dirichlet subsampling.
    """

    def __init__(self, batched_text_dataset: IterableDataset, max_seq_len: int, padding_value: int) -> None:
        data = []
        targets = []
        pad_transform = T.PadTransform(max_length=max_seq_len, pad_value=padding_value)
        for batch in iter(batched_text_dataset):
            data.append(pad_transform(to_tensor(batch["text"], padding_value=padding_value)))
            # the labels of AG_News start from 1 rather than 0, so we subtract 1 from all labels in
            # order to use the cross entropy loss.
            targets.append(torch.tensor(batch["label"]) - 1)
        self.data = torch.cat(data, dim=0)
        self.targets = torch.cat(targets, dim=0)
        self.transform = None
        self.target_transform = None


def construct_dataloaders(
    path: Path,
    batch_size: int,
    max_seq_len: int,
    sample_percentage: float,
    beta: float,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int]]:
    # 256 is the default truncation length used by
    # torchtext's pre-processing pipeline of RoBERTa, and 512 is the maximum
    # sequence length to the RoBERTa-base encoder.
    assert 512 >= max_seq_len >= 256

    # text_transform performs the following steps:
    # 1. Tokenize the sentences using a pre-trained GPT2BPETokenizer.
    # 2. Map the tokenized sentences to RoBERTa's pretrained vocabulary.
    # 3. Truncate the sequences so they have lengths that are at most 256.
    # 4. Add the beginning of sentence and end of sentence tokens.
    text_transform = ROBERTA_BASE_ENCODER.transform()

    # This function simply applies text_transform to every data point.
    def apply_transform(x: Tuple[int, str]) -> Tuple[Union[List[int], List[List[int]]], int]:
        return text_transform(x[1]), x[0]

    ag_datapipe = AG_NEWS(root=path, split="train")
    ag_test_datapipe = AG_NEWS(root=path, split="test")
    data_list = list(ag_datapipe)

    # AG_NEWS does not have a validation set, so we split the training set
    train_size = int(0.95 * len(data_list))
    val_size = int(len(data_list) - train_size)

    train_list_dataset, val_list_dataset = random_split(ListDataset(data_list), [train_size, val_size])

    ag_train_datapipe = IterableWrapper(train_list_dataset)
    ag_val_datapipe = IterableWrapper(val_list_dataset)
    ag_test_datapipe = IterableWrapper(ag_test_datapipe)

    # Tokenize and batch the datasets
    ag_train_datapipe = ag_train_datapipe.map(apply_transform)
    ag_train_datapipe = ag_train_datapipe.batch(batch_size)
    ag_train_datapipe = ag_train_datapipe.rows2columnar(["text", "label"])

    ag_val_datapipe = ag_val_datapipe.map(apply_transform)
    ag_val_datapipe = ag_val_datapipe.batch(batch_size)
    ag_val_datapipe = ag_val_datapipe.rows2columnar(["text", "label"])

    ag_test_datapipe = ag_test_datapipe.map(apply_transform)
    ag_test_datapipe = ag_test_datapipe.batch(batch_size)
    ag_test_datapipe = ag_test_datapipe.rows2columnar(["text", "label"])

    # Unify dataset formats
    ag_train_dataset_full = TextClassificationDataset(ag_train_datapipe, max_seq_len, PAD)
    ag_val_dataset = TextClassificationDataset(ag_val_datapipe, max_seq_len, PAD)
    ag_test_dataset = TextClassificationDataset(ag_test_datapipe, max_seq_len, PAD)

    # Perform Dirchlet sampling on the training set
    sampler = DirichletLabelBasedSampler(list(range(4)), sample_percentage=sample_percentage, beta=beta)
    ag_train_dataset = sampler.subsample(ag_train_dataset_full)

    num_examples = {
        "train_set": len(ag_train_dataset),
        "validation_set": len(ag_val_dataset),
        "test_set": len(ag_test_dataset),
    }

    train_loader: DataLoader = DataLoader(ag_train_dataset, batch_size=batch_size, shuffle=True)
    val_loader: DataLoader = DataLoader(ag_val_dataset, batch_size=batch_size, shuffle=True)
    test_loader: DataLoader = DataLoader(ag_test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader, test_loader, num_examples
