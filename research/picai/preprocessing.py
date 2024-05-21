from __future__ import annotations

import os
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial, reduce
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import SimpleITK as sitk
from preprocessing_transforms import crop_or_pad, resample_img


@dataclass
class PreprocessingSettings:
    def __init__(
        self,
        scans_write_dir: Path,
        annotation_write_dir: Path,
        size: Optional[Sequence[int]],
        physical_size: Optional[Sequence[float]],
        spacing: Optional[Sequence[float]],
    ) -> None:
        """
        Dataclass encapsulating parameters of preprocessing.

        Args:
            scans_write_dir (Path): The directory to write the preprocessed scans.
            annotation_write_dir (Path): The directory to write the preprocessed annotation.
            size (Optional[Sequence[int]]): Sequence of ints representing size of scan in voxels along each dimension.
                If None, preprocessed scans and annotations retain their original size.
            physical_size (Optional[Iterable[float]]): Sequence of floats representing actual size in mm
                along each dimension. If None and size and spacing are not None, physical_size will be inferred.
            spacing (Optional[Sequence[float]]): Spacing between voxels along each dimensions.
                If None, preprocessed scans and annotations retain their original spacing.
        """

        self.scans_write_dir = scans_write_dir
        self.annotation_write_dir = annotation_write_dir
        self.size = size
        self.spacing = spacing

        if physical_size is None and self.spacing is not None and self.size is not None:
            # calculate physical size
            self.physical_size: Sequence[float] = [
                voxel_spacing * num_voxels for voxel_spacing, num_voxels in zip(self.spacing, self.size)
            ]


class Case(ABC):
    def __init__(
        self,
        scan_paths: Sequence[Path],
        annotations_path: Path,
        settings: PreprocessingSettings,
    ) -> None:
        """
        Class representing a case (set of scans along with annotations) along with its preprocessing settings.

        Args:
            scan_paths (Sequence[Path]): The set of paths where the scans associated with the Case are located.
            annotation_write_dir (Path): The path where the annotation associated with the Case is located.
            settings (PreprocessingSettings): The settings determining how the case is preprocessed.
        """
        self.scan_paths = scan_paths
        self.annotations_path = annotations_path
        self.settings = settings

        self.scans: List[sitk.Image]
        self.annotation: sitk.Image

    @abstractmethod
    def read(self) -> None:
        """
        Abstract method to be implemented by children that reads the preprocessed scans and annotation
        into memory.

        Raises:
            NotImplementedError
        """
        raise NotImplementedError

    @abstractmethod
    def write(self) -> Tuple[Sequence[Path], Path]:
        """
        Abstract method to be implemented by children that writes the preprocessed scans and annotation
        to their destination and returns the file paths.

        Returns:
            Tuple[Sequence[Path], Path]: A tuple in which the first entry is a sequence of file paths
                for the scans and the second entry is the file path to the correpsonding annotation.

        Raises:
            NotImplementedError
        """
        raise NotImplementedError


class PicaiCase(Case):
    def __init__(
        self,
        scan_paths: Sequence[Path],
        annotations_path: Path,
        settings: PreprocessingSettings,
    ) -> None:
        """
        Class representing a case from the PICAI dataset.

        Args:
            scan_paths (Sequence[Path]): The set of paths where the scans associated with the Case are located.
            annotation_write_dir (Path): The path where the annotation associated with the Case is located.
            settings (PreprocessingSettings): The settings determining how the case is preprocessed.
        """
        super().__init__(scan_paths, annotations_path, settings)

    def read(self) -> None:
        """
        Reads in scans and annotation into the corresponding scans and annotation attribute.
        """
        assert len(self.scan_paths) != 0
        self.scans = [sitk.ReadImage(path) for path in sorted(self.scan_paths)]
        self.annotation = sitk.ReadImage(self.annotations_path)

    def write(self) -> Tuple[Sequence[Path], Path]:
        """
        Writes preprocessed scans and annotations from PICAI dataset to disk
        and returns the scan file paths and annotation file path in a tuple.

        Returns:
            Tuple[Sequence[Path], Path]: A tuple in which the first entry is a sequence of file paths
                for the scans and the second entry is the file path to the correpsonding annotation.
        """
        modality_suffix_map = {"t2w": "0000", "adc": "0001", "hbv": "0002"}
        scan_paths = [path for path in sorted(self.scan_paths)]
        preprocessed_scan_paths = []
        for path, scan in zip(scan_paths, self.scans):
            scan_filename = os.path.basename(path)
            scan_filename_without_extension = scan_filename.split(".")[0]
            suffix = modality_suffix_map[scan_filename_without_extension[-3:]]
            preprocessed_scan_filename = scan_filename_without_extension[:-3] + suffix + ".nii.gz"
            preprocessed_scan_path = Path(os.path.join(self.settings.scans_write_dir, preprocessed_scan_filename))
            sitk.WriteImage(scan, preprocessed_scan_path, useCompression=True)
            preprocessed_scan_paths.append(preprocessed_scan_path)

        annotation_filename = os.path.basename(self.annotations_path)
        preprocessed_annotation_path = Path(os.path.join(self.settings.annotation_write_dir, annotation_filename))
        sitk.WriteImage(self.annotation, preprocessed_annotation_path, useCompression=True)

        return preprocessed_scan_paths, preprocessed_annotation_path


class PreprocessingException(Exception):
    """
    Custom exception to be thrown during a failed attempt at processing a Case.
    """

    pass


class PreprocessingTransform(ABC):
    """Abstract class that represents a transform to be applied to a case."""

    @abstractmethod
    def __call__(self, mri: Case) -> Case:
        """
        Abstract method to be implemented by children that applies a transformation
        to the input Case and returns the resulting Case.

        Args:
            mri (Case): The case to be processed.

        Returns:
            Case: The Case after the transformation has been applied.
        """
        raise NotImplementedError


class ResampleToFirstScan(PreprocessingTransform):
    def __call__(self, mri: Case) -> Case:
        """
        Resample scans and label to the first scan.

        Args:
            mri (Case): The case to be processed.

        Returns:
            Case: The resampled Case.
        """
        # set up resampler to resolution, field of view, etc. of first scan
        resampler = sitk.ResampleImageFilter()  # default linear
        resampler.SetReferenceImage(mri.scans[0])
        resampler.SetInterpolator(sitk.sitkBSpline)

        # resample other images
        mri.scans[1:] = [resampler.Execute(scan) for scan in mri.scans[1:]]

        # resample annotation
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        mri.annotation = resampler.Execute(mri.annotation)

        return mri


class Resample(PreprocessingTransform):
    def __call__(self, mri: Case) -> Case:
        """
        Resamples scan to a given size.

        Args:
            mri (Case): The case to be processed.

        Returns:
            Case: The resampled Case.
        """
        # resample scans to target resolution
        mri.scans = [resample_img(scan, mri.settings.spacing, is_label=False) for scan in mri.scans]

        # resample annotation to target resolution
        mri.annotation = resample_img(mri.annotation, mri.settings.spacing, is_label=True)

        return mri


class CentreCropAndOrPad(PreprocessingTransform):
    def __call__(self, mri: Case) -> Case:
        """
        Centre crop and/or pad scans and label.

        Args:
            mri (Case): The case to be processed.

        Returns:
            Case: The Case after centre crop and/or padding.
        """
        mri.annotation = crop_or_pad(mri.annotation, mri.settings.size, mri.settings.physical_size)
        mri.scans = [
            crop_or_pad(
                scan,
                mri.settings.size,
                mri.settings.physical_size,
            )
            for scan in mri.scans
        ]

        return mri


class AlignOriginAndDirection(PreprocessingTransform):
    def __call__(self, mri: Case) -> Case:
        """
        Align the origin and direction of each scan, and label.

        Args:
            mri (Case): The case to be processed.

        Returns:
            Case: The Case after aligning origin and direction.
        """
        case_origin, case_direction, case_spacing = None, None, None
        for scan in mri.scans:
            # copy metadata of first scan (nnUNet and nnDetection require this to match exactly)
            if case_origin is None:
                case_origin = scan.GetOrigin()
                case_direction = scan.GetDirection()
                case_spacing = scan.GetSpacing()
            else:
                # check if current scan's metadata is almost equal to the first scan
                np.testing.assert_allclose(scan.GetOrigin(), case_origin)
                np.testing.assert_allclose(scan.GetDirection(), case_direction)
                np.testing.assert_allclose(scan.GetSpacing(), case_spacing)

                # copy over first scan's metadata to current scan
                scan.SetOrigin(case_origin)
                scan.SetDirection(case_direction)
                scan.SetSpacing(case_spacing)

        assert case_origin is not None and case_direction is not None and case_spacing is not None
        mri.annotation.SetOrigin(case_origin)
        mri.annotation.SetDirection(case_direction)
        mri.annotation.SetSpacing(case_spacing)

        return mri


class BinarizeAnnotation(PreprocessingTransform):
    def __call__(self, mri: Case) -> Case:
        """
        Binarize the granular ISUP ≥ 2 annotations.

        Args:
            mri (Case): The case to be processed.

        Returns:
            Case: The Case after binarizing the annotation.
        """
        annotation_array = sitk.GetArrayFromImage(mri.annotation)

        # convert granular PI-CAI csPCa annotation to binary csPCa annotation
        annotation_array = (annotation_array >= 1).astype("uint8")

        # convert label back to SimpleITK
        new_annotation = sitk.GetImageFromArray(annotation_array)
        new_annotation.CopyInformation(mri.annotation)
        mri.annotation = new_annotation
        return mri


def apply_transform(mri: Case, transforms: Sequence[PreprocessingTransform]) -> Tuple[Sequence[Path], Path]:
    """
    Reads in scans and annotation, applies sequence of transformations, and writes resulting case to disk.
    Returns tuple with scan paths and corresponding annotation path.

    Args:
        mri (Case): The case to be processed.
        transforms (Sequence[PreprocessingTransform]): The sequence of transformation to be applied.

    Returns:
        Tuple[Sequence[Path], Path]: A tuple in which the first entry is a sequence of file paths
            for the scans and the second entry is the file path to the correpsonding annotation.

    Raises:
        PreprocessingException if an error occurs during preprocessing.
    """
    try:
        mri.read()
        processed_mri = reduce(lambda acc, f: f(acc), transforms, mri)
        return processed_mri.write()
    except Exception as e:
        error_path_string = ", ".join([str(path) for path in mri.scan_paths] + [str(mri.annotations_path)])
        raise PreprocessingException(f"Error preprocessing case with following paths: {error_path_string}") from e


def preprocess(
    mris: List[Case], transforms: Sequence[PreprocessingTransform], num_threads: int = 4
) -> Sequence[Tuple[Sequence[Path], Path]]:
    """
    Preprocesses a list of cases according to the specified transformations.

    Args:
        mris (List[Case]): A list of cases to be preprocessed.
        transforms (Sequence[PreprocessingTransform]): The sequence of transformation to be applied.
        nums_threads (int): The number of threads to use for preprocessing.

    Returns:
        Sequence[Tuple[Sequence[Path], Path]]: A sequence of tuples in which the first entry is a sequence of
        file paths for the scans and the second entry is the file path to the correpsonding annotation.

    Raises:
        PreprocessingException if an error occurs during preprocessing of any of the cases.
    """
    f = partial(apply_transform, transforms=transforms)
    if num_threads >= 2:
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            return list(executor.map(f, mris))
    else:
        return list(map(f, mris))
