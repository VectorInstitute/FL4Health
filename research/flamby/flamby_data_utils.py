from flamby.datasets.fed_heart_disease import FedHeartDisease
from flamby.datasets.fed_isic2019 import FedIsic2019
from flamby.datasets.fed_ixi import FedIXITiny
from torch.utils.data import random_split


def construct_fedisic_train_val_datasets(
    client_number: int, dataset_dir: str, pooled: bool = False
) -> tuple[FedIsic2019, FedIsic2019]:
    # If pooled is True then client number is ignored
    full_train_dataset = FedIsic2019(center=client_number, train=True, pooled=pooled, data_path=dataset_dir)
    # Something weird is happening with the typing of the split sequence in random split. Punting with a mypy
    # ignore for now.
    train_dataset, validation_dataset = tuple(random_split(full_train_dataset, [0.8, 0.2]))  # type: ignore
    return train_dataset, validation_dataset


def construct_fed_heard_disease_train_val_datasets(
    client_number: int,
    dataset_dir: str,
    pooled: bool = False,
) -> tuple[FedHeartDisease, FedHeartDisease]:
    # If pooled is True then client number is ignored
    full_train_dataset = FedHeartDisease(center=client_number, train=True, pooled=pooled, data_path=dataset_dir)
    # Something weird is happening with the typing of the split sequence in random split. Punting with a mypy
    # ignore for now.
    train_dataset, validation_dataset = tuple(random_split(full_train_dataset, [0.8, 0.2]))  # type: ignore
    return train_dataset, validation_dataset


def construct_fed_ixi_train_val_datasets(
    client_number: int, dataset_dir: str, pooled: bool = False
) -> tuple[FedIXITiny, FedIXITiny]:
    # If pooled is True then client number is ignored
    full_train_dataset = FedIXITiny(
        center=client_number,
        train=True,
        pooled=pooled,
        data_path=dataset_dir,
    )
    # Something weird is happening with the typing of the split sequence in random split. Punting with a mypy
    # ignore for now.
    train_dataset, validation_dataset = tuple(random_split(full_train_dataset, [0.8, 0.2]))  # type: ignore
    return train_dataset, validation_dataset
