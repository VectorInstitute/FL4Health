import argparse
import os
from logging import ERROR, INFO
from pathlib import Path

import numpy as np
import torch
from flwr.common.logger import log
from torch.utils.data import DataLoader
from torchvision import transforms

from fl4health.utils.dataset import TensorDataset
from fl4health.utils.load_data import ToNumpy, get_cifar10_data_and_target_tensors, split_data_and_targets
from fl4health.utils.partitioners import DirichletLabelBasedAllocation
from fl4health.utils.random import set_all_random_seeds


def get_preprocessed_data(
    dataset_dir: Path, client_num: int, batch_size: int, beta: float
) -> tuple[DataLoader, DataLoader, dict[str, int]]:
    transform = transforms.Compose(
        [
            ToNumpy(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    try:
        train_data = torch.from_numpy(np.load(f"{dataset_dir}/beta_{beta}/client_{client_num}_train_data.npy"))
        train_targets = torch.from_numpy(np.load(f"{dataset_dir}/beta_{beta}/client_{client_num}_train_targets.npy"))
    except FileNotFoundError as e:
        log(ERROR, f"Client {client_num} does not have partitioned train data")
        raise e

    training_set = TensorDataset(train_data, train_targets, transform=transform, target_transform=None)

    try:
        validation_data = torch.from_numpy(
            np.load(f"{dataset_dir}/beta_{beta}/client_{client_num}_validation_data.npy")
        )
        validation_targets = torch.from_numpy(
            np.load(f"{dataset_dir}/beta_{beta}/client_{client_num}_validation_targets.npy")
        )
    except FileNotFoundError as e:
        log(ERROR, f"Client {client_num} does not have partitioned validation data")
        raise e

    validation_set = TensorDataset(validation_data, validation_targets, transform=transform, target_transform=None)

    train_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_set, batch_size=batch_size)
    num_examples = {
        "train_set": len(training_set),
        "validation_set": len(validation_set),
    }

    return train_loader, validation_loader, num_examples


def get_test_preprocessed_data(
    dataset_dir: Path, client_num: int, batch_size: int, beta: float
) -> tuple[DataLoader, dict[str, int]]:
    transform = transforms.Compose(
        [
            ToNumpy(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    try:
        data = torch.from_numpy(np.load(f"{dataset_dir}/beta_{beta}/client_{client_num}_test_data.npy"))
        targets = torch.from_numpy(np.load(f"{dataset_dir}/beta_{beta}/client_{client_num}_test_targets.npy"))
    except FileNotFoundError as e:
        log(ERROR, f"Client {client_num} does not have partitioned test data")
        raise e

    evaluation_set = TensorDataset(data, targets, transform=transform, target_transform=None)

    evaluation_loader = DataLoader(evaluation_set, batch_size=batch_size, shuffle=False)
    num_examples = {"eval_set": len(evaluation_set)}

    return evaluation_loader, num_examples


def preprocess_data(
    dataset_dir: Path, num_clients: int, beta: float
) -> tuple[list[TensorDataset], list[TensorDataset], list[TensorDataset]]:
    # Get raw data
    data, targets = get_cifar10_data_and_target_tensors(dataset_dir, True)

    train_data, train_targets, val_data, val_targets = split_data_and_targets(
        data,
        targets,
        validation_proportion=0.1,
    )

    training_set = TensorDataset(train_data, train_targets, transform=None, target_transform=None)
    validation_set = TensorDataset(val_data, val_targets, transform=None, target_transform=None)

    test_data, test_targets = get_cifar10_data_and_target_tensors(dataset_dir, False)
    test_set = TensorDataset(test_data, test_targets, transform=None, target_transform=None)

    # Partition train data
    heterogeneous_partitioner = DirichletLabelBasedAllocation(
        number_of_partitions=num_clients, unique_labels=list(range(10)), beta=beta, min_label_examples=1
    )
    train_partitioned_datasets, train_partitioned_dist = heterogeneous_partitioner.partition_dataset(
        training_set, max_retries=None
    )

    # Partition validation and test data
    heterogeneous_partitioner_with_prior = DirichletLabelBasedAllocation(
        number_of_partitions=num_clients, unique_labels=list(range(10)), prior_distribution=train_partitioned_dist
    )
    validation_partitioned_datasets, _ = heterogeneous_partitioner_with_prior.partition_dataset(
        validation_set, max_retries=None
    )
    test_partitioned_datasets, _ = heterogeneous_partitioner_with_prior.partition_dataset(test_set, max_retries=None)

    return train_partitioned_datasets, validation_partitioned_datasets, test_partitioned_datasets


def save_preprocessed_data(
    save_dataset_dir: Path, partitioned_datasets: list[TensorDataset], beta: float, mode: str
) -> None:
    save_dir_path = f"{save_dataset_dir}/beta_{beta}"
    os.makedirs(save_dir_path, exist_ok=True)

    for client in range(len(partitioned_datasets)):
        save_data = partitioned_datasets[client].data
        save_targets = partitioned_datasets[client].targets
        np.save(f"{save_dir_path}/client_{client}_{mode}_data.npy", save_data.numpy())
        if save_targets is not None:
            np.save(f"{save_dir_path}/client_{client}_{mode}_targets.npy", save_targets.numpy())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FL Client Main")
    parser.add_argument(
        "--dataset_dir",
        action="store",
        type=str,
        help="Path to the raw Cifar 10 Dataset",
        required=True,
    )
    parser.add_argument(
        "--save_dataset_dir",
        action="store",
        type=str,
        help="Path to save the preprocessed Cifar 10 Dataset",
        required=True,
    )
    parser.add_argument(
        "--seed",
        action="store",
        type=int,
        help="Seed for the random number generators across python, torch, and numpy",
        required=False,
    )
    parser.add_argument(
        "--beta",
        action="store",
        type=float,
        help="Heterogeneity level for the dataset",
        required=True,
    )
    parser.add_argument(
        "--num_clients",
        action="store",
        type=int,
        help="Number of clients to partition the dataset into",
        default=5,
    )
    args = parser.parse_args()
    log(INFO, f"Seed: {args.seed}")
    log(INFO, f"Beta: {args.beta}")
    log(INFO, f"Number of clients: {args.num_clients}")

    # Set the random seed for reproducibility
    set_all_random_seeds(args.seed)

    train_partitioned_datasets, validation_partitioned_datasets, test_partitioned_datasets = preprocess_data(
        Path(args.dataset_dir), args.num_clients, args.beta
    )
    save_preprocessed_data(Path(args.save_dataset_dir), train_partitioned_datasets, args.beta, "train")
    save_preprocessed_data(Path(args.save_dataset_dir), validation_partitioned_datasets, args.beta, "validation")
    save_preprocessed_data(Path(args.save_dataset_dir), test_partitioned_datasets, args.beta, "test")
