import json
import os
import random
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
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
            torch.Tensor: Data with moved dimensions.
        """
        data = torch.movedim(data, self.source_dim, self.target_dim)
        return data


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
            torch.Tensor: One hot encoded data.
        """
        return F.one_hot(data.squeeze().long(), num_classes=self.num_classes)


class ZScoreNormalization(Transform):
    """
    MONAI transform to perform z score normalization and optional clipping.
    """

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """
        Z-score normalize input data.

        Args:
            data (torch.Tensor): Input data to be normalized.

        Returns:
            torch.Tensor: Normalized data.
        """
        data = z_score_norm(data)
        return data


def get_img_transform() -> Compose:
    """
    Basic transformation pipeline for images that includes ensuring type and shape of data,
    performing z score normalization, random roation, intensity scaling and adjusting contrast.

    Returns:
        Compose: Image transformation pipeline.
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
        Compose: Segmentation label transformation pipeline.
    """
    transforms = [EnsureType(), EnsureChannelFirst(), MoveDim(-1, 1), OneHotEncode(num_classes=2), MoveDim(-1, 0)]
    return Compose(transforms)


def z_score_norm(image: torch.Tensor, quantile: Optional[float] = None) -> torch.Tensor:
    """
    Function that performs instance wise Z-score normalization (mean=0; stdev=1), where intensities
    below or above the given percentile are discarded.

    Args:
        image (torch.Tensor): N-dimensional image to be normalized and optionally clipped.
        quantile (Optional[float]): Quantile used to set threshold to clip activations.
            If None, no clipping occurs. If a quantile is specified, must be 0 =< 0.5

    Returns:
       torch.Tensor: Z-Score Normalized vesrion of input that is clipped if a quantile is specified.
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
    else:
        return image * 0.0


def get_img_and_seg_paths(
    overviews_dir: Path, base_dir: Path, fold_id: int, train: bool
) -> Tuple[Sequence[Sequence[str]], Sequence[str], torch.Tensor]:
    """
    Gets the image paths, segmentation paths and label proportions for the specified fold.

    Args:
        overviews_dir (Path): A path to the directory containing the marksheets that specify the
            image and segmentation paths for each fold.
        base_dir (Path): The base path of the PICAI dataset.
        fold_id (int): The id of the fold to fetch the image segmentation paths for.
        train (bool): Whether to load the train dataset or the validation dataset.

    Returns:
        Tuple[Sequence[Sequence[str]], Sequence[str], torch.Tensor]: The first element of the returned tuple
            is a list of list of strings where the outer list represents a list of file paths corresponding
            to the diffferent MR Sequences for a given patient exam. The second element is a list of strings
            representing the associated segmentation labels. The final element of the returned tuple is a
            torch tensor that give the class proportions.
    """

    # load datasheets
    file_name = f"PI-CAI_train-fold-{fold_id}.json" if train else f"PI-CAI_val-fold-{fold_id}.json"
    file_path = os.path.join(overviews_dir, file_name)
    with open(Path(file_path)) as fp:
        file_json = json.load(fp)

    # load paths to images and labels
    img_paths = [[os.path.join(base_dir, path) for path in path_list] for path_list in file_json["image_paths"]]
    seg_paths = [os.path.join(base_dir, path) for path in file_json["label_paths"]]

    # Determine class proportions
    class_ratio = [int(np.sum(file_json["case_label"])), int(len(img_paths) - np.sum(file_json["case_label"]))]
    class_proportions = class_ratio / np.sum(class_ratio)

    # Log dataset information
    dataset_name = "Train" if train else "Validation"
    print("Dataset Definition:", "-" * 80)
    print(f"Fold Number: {fold_id}")
    print("Data Classes:", list(np.unique(file_json["case_label"])))
    print(f"{dataset_name} Class Weights: {class_proportions}")
    print(f"{dataset_name} Samples [-:{class_ratio[1]};+:{class_ratio[0]}]: {len(seg_paths)}")

    return img_paths, seg_paths, torch.from_numpy(class_proportions)


def split_img_and_seg_paths(
    img_paths: List[List[str]], seg_paths: List[str], splits: int
) -> Tuple[Sequence[Sequence[Sequence[str]]], Sequence[Sequence[str]]]:
    """
    Split image and segmentation paths into a number of mutually exclusive sets.

    img_paths (Sequence[Sequence[str]]: List of list of strings where the outer list represents
        a list of file paths corresponding to the diffferent MR Sequences for a given patient exam.
    seg_paths (Sequence[str]): List of strings representing the segmentation labels associated with images.
    splits (int): The number of splits to partition the dataset.

    Returns:
        Tuple[Sequence[Sequence[str]], Sequence[str]]: The image and segmentation paths for
        images and segmentation labels.
    """
    assert len(img_paths) == len(seg_paths)

    client_assignments = [random.choice([i for i in range(splits)]) for _ in range(len(img_paths))]
    client_img_paths: List[List[List[str]]] = [[] for _ in range(splits)]
    client_seg_paths: List[List[str]] = [[] for _ in range(splits)]
    for i, assignment in enumerate(client_assignments):
        client_img_paths[assignment].append(img_paths[i])
        client_seg_paths[assignment].append(seg_paths[i])

    return client_img_paths, client_seg_paths


def get_dataloader(
    img_paths: Sequence[Sequence[str]],
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
        img_paths (Sequence[Sequence[str]]: List of list of strings where the outer list represents a
            list of file paths corresponding to the diffferent MR Sequences for a given patient exam.
        seg_paths (Sequence[str]): List of strings representing the segmentation labels associated with images.
        batch_size (str): The number of samples per batch yielded by the DataLoader.
        img_transorm (Compose): The series of transformations applied to input images during dataloading.
        seg_transform (Compose): The series of transformations applied to the segmentation labels during dataloading.
        shuffle (bool): Whether or not to shuffle the dataset.
        num_workers (int): The number of workers used by the DataLoader.

    Returns:
       DataLoader: MONAI dataloader.
    """
    # Ignoring type of image_files because Sequence[Sequence[str]] is valid input
    # list of files interpreted as multi-parametric sequnce. Supported by image loader used by ImageDataset
    # https://docs.monai.io/en/stable/transforms.html#loadimage
    ds = ImageDataset(
        image_files=img_paths,  # type: ignore
        seg_files=seg_paths,
        transform=img_transform,
        seg_transform=seg_transform,
    )

    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return loader
