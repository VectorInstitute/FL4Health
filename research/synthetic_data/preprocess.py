import argparse
import os
from logging import ERROR, INFO
from pathlib import Path

import numpy as np
import torch
from flwr.common.logger import log
from torch.utils.data import DataLoader

from fl4health.utils.data_generation import SyntheticNonIidFedProxDataset
from fl4health.utils.dataset import TensorDataset
from fl4health.utils.load_data import split_data_and_targets
from fl4health.utils.random import set_all_random_seeds


def get_preprocessed_data(
    dataset_dir: Path, client_num: int, batch_size: int, alpha: float, beta: float
) -> tuple[DataLoader, DataLoader, dict[str, int]]:
    try:
        train_data = torch.from_numpy(
            np.load(f"{dataset_dir}/alpha_{alpha}_beta_{beta}/client_{client_num}_train_data.npy")
        ).to(torch.float32)
        train_targets = torch.argmax(
            torch.from_numpy(
                np.load(f"{dataset_dir}/alpha_{alpha}_beta_{beta}/client_{client_num}_train_targets.npy")
            ),
            dim=1,
        )
    except FileNotFoundError as e:
        log(ERROR, f"Client {client_num} does not have partitioned train data")
        raise e

    training_set = TensorDataset(train_data, train_targets, transform=None, target_transform=None)

    try:
        validation_data = torch.from_numpy(
            np.load(f"{dataset_dir}/alpha_{alpha}_beta_{beta}/client_{client_num}_validation_data.npy")
        ).to(torch.float32)
        validation_targets = torch.argmax(
            torch.from_numpy(
                np.load(f"{dataset_dir}/alpha_{alpha}_beta_{beta}/client_{client_num}_validation_targets.npy")
            ),
            dim=1,
        )
    except FileNotFoundError as e:
        log(ERROR, f"Client {client_num} does not have partitioned validation data")
        raise e

    validation_set = TensorDataset(validation_data, validation_targets, transform=None, target_transform=None)

    train_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_set, batch_size=batch_size)
    num_examples = {
        "train_set": len(training_set),
        "validation_set": len(validation_set),
    }

    return train_loader, validation_loader, num_examples


def get_test_preprocessed_data(
    dataset_dir: Path, client_num: int, batch_size: int, alpha: float, beta: float
) -> tuple[DataLoader, dict[str, int]]:
    try:
        data = torch.from_numpy(
            np.load(f"{dataset_dir}/alpha_{alpha}_beta_{beta}/client_{client_num}_test_data.npy")
        ).to(torch.float32)
        targets = torch.argmax(
            torch.from_numpy(np.load(f"{dataset_dir}/alpha_{alpha}_beta_{beta}/client_{client_num}_test_targets.npy")),
            dim=1,
        )
    except FileNotFoundError as e:
        log(ERROR, f"Client {client_num} does not have partitioned test data)")
        raise e

    evaluation_set = TensorDataset(data, targets, transform=None, target_transform=None)

    evaluation_loader = DataLoader(evaluation_set, batch_size=batch_size, shuffle=False)
    num_examples = {"eval_set": len(evaluation_set)}

    return evaluation_loader, num_examples


def preprocess_data(
    alpha: float,
    beta: float,
    num_clients: int = 5,
) -> tuple[list[TensorDataset], list[TensorDataset], list[TensorDataset]]:
    # Get raw data
    synth_data_generator = SyntheticNonIidFedProxDataset(
        num_clients=num_clients,
        alpha=alpha,
        beta=beta,
        temperature=2.0,
        input_dim=60,
        output_dim=10,
        hidden_dim=20,
        samples_per_client=5000,
    )

    partitioned_client_datasets = synth_data_generator.generate()

    train_partitioned_datasets = []
    validation_partitioned_datasets = []
    test_partitioned_datasets = []
    for client_dataset in partitioned_client_datasets:
        assert client_dataset.targets is not None
        data, targets = client_dataset.data, client_dataset.targets
        train_data, train_targets, test_data, test_targets = split_data_and_targets(
            data,
            targets,
            validation_proportion=0.2,
        )
        train_data, train_targets, val_data, val_targets = split_data_and_targets(
            train_data,
            train_targets,
            validation_proportion=0.2,
        )

        training_set = TensorDataset(train_data, train_targets, transform=None, target_transform=None)
        train_partitioned_datasets.append(training_set)
        validation_set = TensorDataset(val_data, val_targets, transform=None, target_transform=None)
        validation_partitioned_datasets.append(validation_set)
        test_set = TensorDataset(test_data, test_targets, transform=None, target_transform=None)
        test_partitioned_datasets.append(test_set)

    return train_partitioned_datasets, validation_partitioned_datasets, test_partitioned_datasets


def save_preprocessed_data(
    save_dataset_dir: Path, partitioned_datasets: list[TensorDataset], alpha: float, beta: float, mode: str
) -> None:
    save_dir_path = f"{save_dataset_dir}/alpha_{alpha}_beta_{beta}"
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
        "--alpha",
        action="store",
        type=float,
        help="Heterogeneity parameter to generate the elements of the affine transformation for the dataset",
        required=True,
    )
    parser.add_argument(
        "--beta",
        action="store",
        type=float,
        help="Heterogeneity parameter to generate the elements of the input features of the dataset",
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
    log(INFO, f"Alpha: {args.alpha}")
    log(INFO, f"Beta: {args.beta}")
    log(INFO, f"Number of clients: {args.num_clients}")

    # Set the random seed for reproducibility
    set_all_random_seeds(args.seed)

    train_partitioned_datasets, validation_partitioned_datasets, test_partitioned_datasets = preprocess_data(
        args.alpha,
        args.beta,
        args.num_clients,
    )
    save_preprocessed_data(Path(args.save_dataset_dir), train_partitioned_datasets, args.alpha, args.beta, "train")
    save_preprocessed_data(
        Path(args.save_dataset_dir), validation_partitioned_datasets, args.alpha, args.beta, "validation"
    )
    save_preprocessed_data(Path(args.save_dataset_dir), test_partitioned_datasets, args.alpha, args.beta, "test")
