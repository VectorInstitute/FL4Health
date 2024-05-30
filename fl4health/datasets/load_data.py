import json
import os
import random
from logging import INFO
from typing import Callable, Dict, Optional, Tuple, Union

import torch
import torchvision.transforms as transforms
from flwr.common.logger import log
from torch.utils.data import DataLoader, RandomSampler

from fl4health.datasets.fairseq_signals.data import FileECGDataset
from fl4health.utils.dataset import BaseDataset
from fl4health.utils.dataset_converter import DatasetConverter
from fl4health.utils.sampler import LabelBasedSampler


def load_skin_cancer_data(
    dataset_name: str,
    batch_size: int,
    sampler: Optional[LabelBasedSampler] = None,
    transform: Union[None, Callable] = None,
    target_transform: Union[None, Callable] = None,
    dataset_converter: Optional[DatasetConverter] = None,
    seed: int = 0,
) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
    """Load skin cancer dataset (training, validation, and test set)."""
    common_path = "fl4health/utils/datasets"

    dataset_paths = {
        "Barcelona": os.path.join(common_path, "ISIC_2019", "ISIC_19_Barcelona.json"),
        "Rosendahl": os.path.join(common_path, "HAM10000", "HAM_rosendahl.json"),
        "Vienna": os.path.join(common_path, "HAM10000", "HAM_vienna.json"),
        "UFES": os.path.join(common_path, "PAD-UFES-20", "PAD_UFES_20.json"),
        "Canada": os.path.join(common_path, "Derm7pt", "Derm7pt.json"),
    }

    dataset_path = dataset_paths[dataset_name]
    log(INFO, f"Data directory: {str(dataset_path)}")

    with open(dataset_path, "r") as f:
        data = json.load(f)["data"]

    random.seed(seed)
    random.shuffle(data)
    total_size = len(data)
    train_size = int(0.7 * total_size)
    valid_size = int(0.15 * total_size)

    train_data = data[:train_size]
    valid_data = data[train_size : train_size + valid_size]

    if transform is None:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    train_ds: BaseDataset = BaseDataset(train_data, transform=transform, target_transform=target_transform)
    valid_ds: BaseDataset = BaseDataset(valid_data, transform=transform, target_transform=target_transform)

    if sampler is not None:
        train_ds = sampler.subsample(train_ds)
        valid_ds = sampler.subsample(valid_ds)

    if dataset_converter is not None:
        train_ds = dataset_converter.convert_dataset(train_ds)
        valid_ds = dataset_converter.convert_dataset(valid_ds)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(valid_ds, batch_size=batch_size)

    num_examples = {"train_set": len(train_ds), "validation_set": len(valid_ds)}

    return train_loader, validation_loader, num_examples


def load_skin_cancer_test_data(
    dataset_name: str,
    batch_size: int,
    sampler: Optional[LabelBasedSampler] = None,
    transform: Union[None, Callable] = None,
    target_transform: Union[None, Callable] = None,
) -> Tuple[DataLoader, Dict[str, int]]:
    """Load skin cancer dataset (test set only)."""
    common_path = "fl4health/utils/datasets"

    dataset_paths = {
        "Barcelona": os.path.join(common_path, "ISIC_2019", "ISIC_19_Barcelona.json"),
        "Rosendahl": os.path.join(common_path, "HAM10000", "HAM_rosendahl.json"),
        "Vienna": os.path.join(common_path, "HAM10000", "HAM_vienna.json"),
        "UFES": os.path.join(common_path, "PAD-UFES-20", "PAD_UFES_20.json"),
        "Canada": os.path.join(common_path, "Derm7pt", "Derm7pt.json"),
    }

    dataset_path = dataset_paths[dataset_name]
    log(INFO, f"Data directory: {str(dataset_path)}")

    with open(dataset_path, "r") as f:
        data = json.load(f)["data"]

    random.seed(0)  # For consistency
    random.shuffle(data)
    total_size = len(data)
    test_size = int(0.15 * total_size)
    test_data = data[-test_size:]

    if transform is None:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    test_ds: BaseDataset = BaseDataset(test_data, transform=transform, target_transform=target_transform)

    if sampler is not None:
        test_ds = sampler.subsample(test_ds)

    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    num_examples = {"test_set": len(test_ds)}
    return test_loader, num_examples


def get_split_loader(args, path, split=True):
    split_dataset = FileECGDataset(
        manifest_path=path,
        sample_rate=500,
        max_sample_size=None,
        min_sample_size=None,
        pad=True,
        pad_leads=False,
        leads_to_load=None,
        label=True,
        normalize=False,
        num_buckets=0,
        compute_mask_indices=False,
        leads_bucket=None,
        bucket_selection="uniform",
        training=split,  # True, False
        **{},
    )

    if split == True:
        sampler = RandomSampler(split_dataset)
    else:
        sampler = None

    data_size = len(split_dataset)

    data_loader = torch.utils.data.DataLoader(
        split_dataset,
        batch_size=args.batch_size,
        collate_fn=split_dataset.collator,
        sampler=sampler,
        num_workers=args.num_workers,
    )

    return data_loader, data_size


def get_dataloader(args):

    train_loaders, valid_loaders, test_loaders = [], [], []
    client_weights = []

    for dataname in args.data_list:
        common_dir = f"{args.load_dir}/{dataname}/cinc"
        train_loader, train_size = get_split_loader(args, os.path.join(common_dir, "train.tsv"), split=True)
        valid_loader, valid_size = get_split_loader(args, os.path.join(common_dir, "valid.tsv"), split=False)
        test_loader, test_size = get_split_loader(args, os.path.join(common_dir, "test.tsv"), split=False)

        train_loaders.append(train_loader)
        valid_loaders.append(valid_loader)
        test_loaders.append(test_loader)
        client_weights.append(train_size)

    client_weighted = [c_weight / sum(client_weights) for c_weight in client_weights]

    return train_loaders, valid_loaders, test_loaders, client_weighted
