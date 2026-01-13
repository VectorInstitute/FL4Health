import json
import os
import random
from collections.abc import Sequence
from logging import INFO
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from flwr.common.logger import log
from monai.data.dataloader import DataLoader
from monai.data.image_dataset import ImageDataset
from monai.transforms import Transform
from monai.transforms.compose import Compose
from monai.transforms.intensity.array import AdjustContrast, ScaleIntensity
from monai.transforms.utility.array import EnsureChannelFirst, EnsureType


class MoveDim(Transform):
    def __init__(self, source_dim: int, target_dim: int) -> None:
        """
        MONAI transform that moves source dimension to a target dimension.

        Args:
            source_dim (int): The index of the source dimension.
            target_dim (int): The index of the target dimension.
        """
        self.source_dim = source_dim
        self.target_dim = target_dim
        super().__init__()

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """
        Moves data's source dim to target dim.

        Args:
            data (torch.Tensor): Data to be transformed.

        Returns:
            (torch.Tensor): Data with moved dimensions.
        """
        return torch.movedim(data, self.source_dim, self.target_dim)


class OneHotEncode(Transform):
    def __init__(self, num_classes: int = 2) -> None:
        """
        MONAI transform to one hot encode labels.

        Args:
            num_classes (int): The number of classes to encode.
        """
        self.num_classes = num_classes
        super().__init__()

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """
        One hot encode input data.

        Args:
            data (torch.Tensor): Data to be transformed.

        Returns:
            (torch.Tensor): One hot encoded data.
        """
        return F.one_hot(data.squeeze().long(), num_classes=self.num_classes)


class ZScoreNormalization(Transform):
    """MONAI transform to perform z score normalization and optional clipping."""

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """
        Z-score normalize input data.

        Args:
            data (torch.Tensor): Input data to be normalized.

        Returns:
            (torch.Tensor): Normalized data.
        """
        return z_score_norm(data)


def get_img_transform() -> Compose:
    """
    Basic transformation pipeline for images that includes ensuring type and shape of data,
    performing z score normalization, random rotation, intensity scaling and adjusting contrast.

    Returns:
        (Compose): Image transformation pipeline.
    """
    transforms = [
        EnsureType(),
        EnsureChannelFirst(),
        ZScoreNormalization(),
        ScaleIntensity(minv=0.7, maxv=1.4),
        AdjustContrast(gamma=1.0),
        MoveDim(-1, 1),
    ]
    return Compose(transforms)


def get_seg_transform() -> Compose:
    """
    Basic transformation pipeline for labels that includes ensuring type and shape of data,
    along with One-Hot-Encoding.

    Returns:
        (Compose): Segmentation label transformation pipeline.
    """
    transforms = [EnsureType(), EnsureChannelFirst(), MoveDim(-1, 1), OneHotEncode(num_classes=2), MoveDim(-1, 0)]
    return Compose(transforms)


def z_score_norm(image: torch.Tensor, quantile: float | None = None) -> torch.Tensor:
    """
    Function that performs instance wise Z-score normalization (mean=0; stdev=1), where intensities
    below or above the given percentile are discarded.

    Args:
        image (torch.Tensor): N-dimensional image to be normalized and optionally clipped.
        quantile (float | None): Quantile used to set threshold to clip activations.
            If None, no clipping occurs. If a quantile is specified, must be 0 =< 0.5

    Returns:
       (torch.Tensor): Z-Score Normalized version of input that is clipped if a quantile is specified.
    """
    image = image.float()

    if quantile is not None:
        assert quantile >= 0.0 and quantile <= 0.5
        # clip distribution of intensity values
        lower_bnd = torch.quantile(image, 1.0 - quantile)
        upper_bnd = torch.quantile(image, quantile)
        image = torch.clip(image, lower_bnd, upper_bnd)

    # perform z-score normalization
    mean = torch.mean(image)
    std = torch.std(image).item()
    if std > 0:
        return (image - mean) / std
    return image * 0.0


def get_img_and_seg_paths(
    overviews_dir: Path,
    fold_id: int,
    train: bool,
    include_t2w: bool = True,
    include_adc: bool = True,
    include_hbv: bool = True,
) -> tuple[list[list[str]], list[str], torch.Tensor]:
    """
    Gets the image paths, segmentation paths and label proportions for the specified fold.
    Exclude t2w, adc or hbv scan if specified.

    Args:
        overviews_dir (Path): A path to the directory containing the marksheets that specify the
            image and segmentation paths for each fold.
        base_dir (Path): The base path of the PICAI dataset.
        fold_id (int): The id of the fold to fetch the image segmentation paths for.
        train (bool): Whether to load the train dataset or the validation dataset.
        include_t2w (bool): Whether or not to include t2w Sequence as part of the input data.
        include_adc (bool): Whether or not to include adc Sequence as part of the input data.
        include_hbv (bool): Whether or not to include hbv Sequence as part of the input data.

    Returns:
        (tuple[Sequence[Sequence[str]], Sequence[str], torch.Tensor]): The first element of the returned tuple
            is a list of list of strings where the outer list represents a list of file paths corresponding
            to the different MRI Sequences for a given patient exam. The second element is a list of strings
            representing the associated segmentation labels. The final element of the returned tuple is a
            torch tensor that give the class proportions.
    """
    # Make sure at least one sequence will be present
    assert any([include_t2w, include_adc, include_hbv])

    # load dataset overview
    file_name = f"train-fold-{fold_id}.json" if train else f"val-fold-{fold_id}.json"
    file_path = os.path.join(overviews_dir, file_name)
    with open(Path(file_path)) as fp:
        file_json = json.load(fp)

    # Determine valid file name endings (indicator for sequence type: T1, T2, ADC)
    all_postfix = ("0000.nii.gz", "0001.nii.gz", "0002.nii.gz")
    valid_postfix = tuple([item for flag, item in zip([include_t2w, include_adc, include_hbv], all_postfix) if flag])

    # load paths to images and labels
    img_paths = [
        [path for path in path_list if path.endswith(valid_postfix)] for path_list in file_json["image_paths"]
    ]
    seg_paths = list(file_json["label_paths"])

    # Determine class proportions
    class_ratio = [int(np.sum(file_json["case_label"])), int(len(img_paths) - np.sum(file_json["case_label"]))]
    class_proportions = class_ratio / np.sum(class_ratio)

    dataset_type_string = "Train" if train else "Validation"
    log(
        INFO,
        f"Dataset Type: {dataset_type_string} "
        f"Fold ID: {str(fold_id)} "
        f"Num Samples: {len(seg_paths)} "
        f"Data Classes: {str(np.unique(file_json['case_label']))} "
        f"Class Proportions: {str(np.unique(file_json['case_label']))}",
    )

    return img_paths, seg_paths, torch.from_numpy(class_proportions)


def split_img_and_seg_paths(
    img_paths: list[list[str]], seg_paths: list[str], splits: int, seed: int = 0
) -> tuple[list[list[list[str]]], list[list[str]]]:
    """
    Split image and segmentation paths into a number of mutually exclusive sets.

    img_paths (Sequence[Sequence[str]]: List of list of strings where the outer list represents
        a list of file paths corresponding to the different MRI Sequences for a given patient exam.
    seg_paths (Sequence[str]): List of strings representing the segmentation labels associated with images.
    splits (int): The number of splits to partition the dataset.

    Returns:
        (tuple[Sequence[Sequence[str]], Sequence[str]]): The image and segmentation paths for
        images and segmentation labels.
    """
    assert len(img_paths) == len(seg_paths)
    random.seed(seed)
    client_assignments = [random.choice(list(range(splits))) for _ in range(len(img_paths))]
    client_img_paths: list[list[list[str]]] = [[] for _ in range(splits)]
    client_seg_paths: list[list[str]] = [[] for _ in range(splits)]
    for i, assignment in enumerate(client_assignments):
        client_img_paths[assignment].append(img_paths[i])
        client_seg_paths[assignment].append(seg_paths[i])

    return client_img_paths, client_seg_paths


def get_dataloader(
    img_paths: Sequence[Sequence[str]] | Sequence[str],
    seg_paths: Sequence[str],
    batch_size: int,
    img_transform: Compose,
    seg_transform: Compose,
    shuffle: bool = False,
    num_workers: int = 2,
) -> DataLoader:
    """
    Initializes and returns MONAI Dataloader.

    Args:
        img_paths (Sequence[Sequence[str]] | Sequence[str]): List of list of strings where the outer list represents a
            list of file paths corresponding to the different MRI Sequences for a given patient exam.
        seg_paths (Sequence[str]): List of strings representing the segmentation labels associated with images.
        batch_size (int): The number of samples per batch yielded by the DataLoader.
        img_transform (Compose): The series of transformations applied to input images during dataloading.
        seg_transform (Compose): The series of transformations applied to the segmentation labels during dataloading.
        shuffle (bool, optional): Whether or not to shuffle the dataset. Defaults to False.
        num_workers (int, optional): The number of workers used by the DataLoader. Defaults to 2.

    Returns:
        (DataLoader): MONAI dataloader.
    """
    # Ignoring type of image_files because Sequence[Sequence[str]] is valid input
    # list of files interpreted as multi-parametric sequence. Supported by image loader:
    # https://docs.monai.io/en/stable/transforms.html#loadimage used by ImageDataset:
    # https://docs.monai.io/en/latest/data.html#monai.data.ImageDataset
    ds = ImageDataset(
        image_files=img_paths,  # type: ignore
        seg_files=seg_paths,
        transform=img_transform,
        seg_transform=seg_transform,
    )

    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
