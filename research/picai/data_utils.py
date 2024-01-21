import os
import random
import json
from pathlib import Path
from typing import Callable, Optional, Tuple, Sequence
import numpy as np
import torch
import torch.nn.functional as F
from monai.transforms import Transform
from monai.transforms.compose import Compose
from monai.transforms.utility.array import EnsureChannelFirst, EnsureType
from monai.transforms import RandRotate
from monai.transforms.intensity.array import AdjustContrast, ScaleIntensity
from monai.data.image_dataset import ImageDataset
from monai.data.dataloader import DataLoader

augmentation_params = {
    "scale_range": (0.7, 1.4),
    "gamma": 1.0
}


class MoveDim(Transform):
    def __init__(self, source_dim: int, target_dim: int) -> None:
        self.source_dim = source_dim
        self.target_dim = target_dim
        super().__init__()

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        data = torch.movedim(data, self.source_dim, self.target_dim)
        return data


class OneHotEncode(Transform):
    def __init__(self, num_classes: int = 2) -> None:
        self.num_classes = num_classes
        super().__init__()

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        return F.one_hot(data.squeeze().long(), num_classes=self.num_classes)


class ZScoreNormalization(Transform):
    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        data = z_score_norm(data)
        return data


def get_img_transform() -> Compose:
    transforms = [
        EnsureType(),
        EnsureChannelFirst(),
        ZScoreNormalization(),
        RandRotate(),
        ScaleIntensity(minv=augmentation_params["scale_range"][0], maxv=augmentation_params["scale_range"][1]),
        AdjustContrast(gamma=1.0),
        MoveDim(-1, 1)
    ]
    return Compose(transforms)


def get_seg_transform() -> Compose:
    transforms = [
        EnsureType(),
        EnsureChannelFirst(),
        MoveDim(-1, 1),
        OneHotEncode(num_classes=2),
        MoveDim(-1, 0)
    ]
    return Compose(transforms)


def z_score_norm(image: torch.Tensor, quantile: Optional[float] = None) -> torch.Tensor:
    """
    Function that performs instance wise Z-score normalization (mean=0; stdev=1), where intensities
    below or above the given percentile are discarded.

    Args:
        image (npt.Ndarray[Any]): N-dimensional image to be normalized and optionally clipped. 
        percentile (Optional[float]): Percentile used to set threshold to clip activations.
            If None, no clipping occurs. If a percentile is specified, must be 0 =< 50

    Returns:
       npt.NDArray[Any]: Z-Score Normalized vesrion of input that is clipped if a percentile is specified. 
    """
    image = image.float()

    if quantile is not None:
        assert (quantile >= 0.0 and quantile <= 0.5)
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
        return image * 0.


def get_img_and_seg_paths(overviews_dir : str, base_dir: str, fold_id: int, train: bool) -> Tuple[Sequence[Sequence[str]], Sequence[str], torch.Tensor]:

    # load datasheets
    file_name = f"PI-CAI_train-fold-{fold_id}.json" if train else f"PI-CAI_val-fold-{fold_id}.json"
    file_path = os.path.join(overviews_dir, file_name)
    with open(Path(file_path)) as fp:
        file_json = json.load(fp)

    # load paths to images and labels
    img_paths = [[os.path.join(base_dir, path) for path in path_list] for path_list in file_json["image_paths"]]
    seg_paths = [os.path.join(base_dir, path) for path in file_json["label_paths"]]

    # Determine class proportions
    class_ratio = [int(np.sum(file_json['case_label'])), int(len(img_paths) - np.sum(file_json['case_label']))]
    class_proportions = (class_ratio / np.sum(class_ratio))

    # Log dataset information
    dataset_name = "Train" if train else "Validation"
    print('Dataset Definition:', "-" * 80)
    print(f'Fold Number: {fold_id}')
    print('Data Classes:', list(np.unique(file_json['case_label'])))
    print(f'{dataset_name} Class Weights: {class_proportions}')
    print(f'{dataset_name} Samples [-:{class_ratio[1]};+:{class_ratio[0]}]: {len(seg_paths)}')

    return img_paths, seg_paths, torch.from_numpy(class_proportions)


def split_img_and_seg_paths(img_paths: Sequence[Sequence[str]], seg_paths: Sequence[str], splits: int) -> Tuple[Sequence[Sequence[Sequence[str]]], Sequence[Sequence[str]]]:
    assert len(img_paths) == len(seg_paths)

    client_assignments = [random.choice([i for i in range(splits)]) for _ in range(len(img_paths))]
    client_img_paths = [[] for _ in range(splits)]
    client_seg_paths = [[] for _ in range(splits)]
    for i, assignment in enumerate(client_assignments):
        client_img_paths[assignment].append(img_paths[i])
        client_seg_paths[assignment].append(seg_paths[i])

    return client_img_paths, client_seg_paths


def get_dataloader(img_paths: Sequence[Sequence[str]], seg_paths: Sequence[str], batch_size: int, img_transform: Callable, seg_transform: Callable, shuffle: bool = False, num_workers: int =2) -> DataLoader:
    """
    Function that initializes and returns the train and validation DataLoader along with proportion of samples
    with each label.

    Args:
        overview_dir (str): The path to the directory that houses the datasheets for the train and validation data.
        batch_size (str): The number of samples per batch yielded by the DataLoader.
        num_threads (int): The number of threads used by the DataLoader. 
        fold_id (int): The cross validation fold to use to create the DataLoader.

    Returns:
       Tuple[DataLoader, DataLoader, np.array]: The Training DataLoader, Validation Loader and Numpy Array
       with the proportion of samples in each class.

    """

    ds = ImageDataset(
        image_files=img_paths,
        seg_files=seg_paths,
        transform=img_transform,
        seg_transform=seg_transform,
    )

    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return loader
