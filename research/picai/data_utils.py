import os
import json
from pathlib import Path
from typing import Any, Optional, Tuple
import numpy as np
import torch
from monai.transforms import Transform
from monai.transforms.compose import Compose
from monai.transforms.utility.array import EnsureChannelFirst, EnsureType
from monai.transforms import RandRotate
from monai.transforms.intensity.array import AdjustContrast, ScaleIntensity
from monai.data.image_dataset import ImageDataset
from monai.data.dataloader import DataLoader

augmentation_params = {
    "rotation_x": (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
    "rotation_y": (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
    "rotation_z": (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
    "p_rot": 0.2,
    "scale_range": (0.7, 1.4),
    "independent_scale_factor_for_each_axis": False,
    "p_scale": 0.2,
    "gamma_retain_stats": True,
    "gamma_range": (0.7, 1.5),
    "p_gamma": 0.3,
}


class MoveDim(Transform):
    def __init__(self, source_dim: int, target_dim: int) -> None:
        self.source_dim = source_dim
        self.target_dim = target_dim

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        data = torch.movedim(data, self.source_dim, self.target_dim)
        return data


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
        MoveDim(-1, 1)
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


def get_dataloaders(overview_dir: str, base_dir: str, batch_size: int, num_threads: int, fold_id: int) -> Tuple[DataLoader, DataLoader, "npt.NDArray[np.float32]"]:
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
    # load datasheets
    with open(Path(overview_dir) / f'PI-CAI_train-fold-{fold_id}.json') as fp:
        train_json = json.load(fp)
    with open(Path(overview_dir) / f'PI-CAI_val-fold-{fold_id}.json') as fp:
        val_json = json.load(fp)
    # load paths to images and labels
    train_img_paths = [[os.path.join(base_dir, path) for path in path_list] for path_list in train_json["image_paths"]]
    train_seg_paths = [os.path.join(base_dir, path) for path in train_json["label_paths"]]
    val_img_paths = [[os.path.join(base_dir, path) for path in path_list] for path_list in val_json["image_paths"]]
    val_seg_paths = [os.path.join(base_dir, path) for path in val_json["label_paths"]]
    train_data = [train_img_paths, train_seg_paths]
    val_data = [val_img_paths, val_seg_paths]

    # use case-level class balance to deduce required train-time class weights
    class_ratio_t = [int(np.sum(train_json['case_label'])), int(len(train_data[0]) - np.sum(train_json['case_label']))]
    class_ratio_v = [int(np.sum(val_json['case_label'])), int(len(val_data[0]) - np.sum(val_json['case_label']))]
    class_proportions = (class_ratio_t / np.sum(class_ratio_t))

    # log dataset definition
    print('Dataset Definition:', "-" * 80)
    print(f'Fold Number: {fold_id}')
    print('Data Classes:', list(np.unique(train_json['case_label'])))
    print(f'Train-Time Class Weights: {class_proportions}')
    print(f'Training Samples [-:{class_ratio_t[1]};+:{class_ratio_t[0]}]: {len(train_data[1])}')
    print(f'Validation Samples [-:{class_ratio_v[1]};+:{class_ratio_v[0]}]: {len(val_data[1])}')

    train_transform_img = get_img_transform()
    train_transform_seg = get_seg_transform()
    train_ds = ImageDataset(
        image_files=train_data[0],
        seg_files=train_data[1],
        transform=train_transform_img,
        seg_transform=train_transform_seg,
    )

    val_transform_img = get_img_transform()
    val_transform_seg = get_seg_transform()
    val_ds = ImageDataset(
        image_files=val_data[0],
        seg_files=val_data[1],
        transform=val_transform_img,
        seg_transform=val_transform_seg,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader, torch.from_numpy(class_proportions)
