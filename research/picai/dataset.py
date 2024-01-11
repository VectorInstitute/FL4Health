from typing import Any, Callable, Optional, Sequence
import numpy as np
from torch.utils.data import Dataset
import SimpleITK as sitk
from monai.config.type_definitions import DtypeLike
from monai.transforms.transform import Randomizable, apply_transform
from monai.utils.misc import MAX_SEED, get_seed
from research.picai.data_utils import z_score_norm
from numpy import typing as npt


class SimpleITKDataset(Dataset, Randomizable):
    """
    """

    def __init__(
        self,
        image_files: Sequence[str],
        seg_files: Optional[Sequence[str]] = None,
        labels: Optional[Sequence[float]] = None,
        transform: Optional[Callable] = None,
        seg_transform: Optional[Callable] = None,

    ) -> None:
        """
        Loads image/segmentation pairs of files from the given filename lists. Transformations canbe specified for the image and segmentation arrays separately. The difference between thisdataset and `ArrayDataset` is that this dataset can apply transform chain to images
        and segs and return both the images and metadata, and no need to specify transform to loadimages from files. For more information, please see the image_dataset demo in the MONAI tutorial repo, https://github.com/Project-MONAI/tutorials/blob/master/modules/image_dataset.ipynb. Also performs instance-wise z-score normalization of all MRI sequences before concatenation. Initializes the dataset with the image and segmentation filename lists. The transform `transform` is applied to the images and `seg_transform` to the segmentations.

        Args:
            image_files (Sequence[str]): List of image filenames.
            seg_files (Optional[Sequence[str]]): List of segmentation filenames.
                If not None, presume segmentation task.
            labels (Optional[Sequence[float]]: List of classification labels.
                If not None, presume classification task.
            transform (Callable): Transform to apply to image arrays.
            seg_transform (Callable): Transform to apply to segmentation arrays

        Raises:
            ValueError: When seg_files length differs from image_files.

        """

        if seg_files is not None and len(image_files) != len(seg_files):
            raise ValueError(
                "Must have same the number of segmentation as image files: "
                f"images={len(image_files)}, segmentations={len(seg_files)}."
            )

        self.image_files = image_files
        self.seg_files = seg_files
        self.labels = labels
        self.transform = transform
        self.seg_transform = seg_transform
        self.set_random_state(seed=get_seed())
        self._seed = 0  # transform synchronization seed

    def __len__(self) -> int:
        """
        Returns length of the dataset.

        Returns:
            int: Length of dataset.
        """
        return len(self.image_files)

    def randomize(self, data: Optional[Any] = None) -> None:
        """
        Samples and sets random seed for Dataset.

        Args:
            data (Optional[Any]]): Unused argument for compatibiliy
                with Randomizable whih this class extends.
        """
        self._seed = self.R.randint(MAX_SEED, dtype="uint32")

    def prepare_scan(self, path: str) -> "npt.NDArray[Any]":
        """
        Loads ITK image given by path into a Numpy Array with desired
        type and dimensionality.

        Args:
            path (str): Path to the ITK data.

        Returns:
            "npt.NDArray[Any]": Numpy Array representing ITK Image.
        """
        return np.expand_dims(
            sitk.GetArrayFromImage(
                sitk.ReadImage(path)
            ).astype(np.float32), axis=(0, 1)
        )

    # TODO: Pin down return type once accepted types have been established. 
    def __getitem__(self, index: int) -> Any:
        """
        Loads data, and label if applicable, corresponding to the index argument,
        applies transformations and returns sample.

        Args:
            index (int): Index of sample to retrieve.

        Returns:
            Any: Sample with optional corresponding segmentation mask and label.
        """
        self.randomize()
        seg, label = None, None

        # load all sequences (work-around) and optionally meta
        img_t2w = z_score_norm(self.prepare_scan(str(self.image_files[index][0])), 99.5)
        img_adc = z_score_norm(self.prepare_scan(str(self.image_files[index][1])), 99.5)
        img_hbv = z_score_norm(self.prepare_scan(str(self.image_files[index][2])), 99.5)

        img = np.concatenate([img_t2w, img_adc, img_hbv], axis=1)

        if self.seg_files is not None:
            seg = sitk.GetArrayFromImage(sitk.ReadImage(self.seg_files[index])).astype(np.int8)
            seg = np.expand_dims(seg, axis=(0, 1))

        # apply the transforms
        if self.transform is not None:
            img = apply_transform(self.transform, img, map_items=False)

        if self.seg_transform is not None:
            seg = apply_transform(self.seg_transform, seg, map_items=False)

        if self.labels is not None:
            label = self.labels[index]

        # construct outputs
        data = [img]
        if seg is not None:
            data.append(seg)
        if label is not None:
            data.append(label)
        if len(data) == 1:
            return data[0]

        # use tuple instead of list as the default collate_fn callback of MONAI DataLoader flattens nested lists
        return tuple(data)
