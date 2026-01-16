from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial, reduce
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from research.picai.data.preprocessing_transforms import crop_or_pad, resample_img


@dataclass
class PreprocessingSettings:
    def __init__(
        self,
        scans_write_dir: Path,
        annotation_write_dir: Path,
        size: tuple[int, int, int] | None,
        physical_size: tuple[float, float, float] | None,
        spacing: tuple[float, float, float] | None,
    ) -> None:
        """
        Dataclass encapsulating parameters of preprocessing.

        Args:
            scans_write_dir (Path): The directory to write the preprocessed scans.
            annotation_write_dir (Path): The directory to write the preprocessed annotation.
            size (tuple[int, int, int] | None): Tuple of 3 int representing size of scan in voxels.
                In the format of Depth x Height x Width. If None, preprocessed scans and annotations retain
                their original size.
            physical_size (tuple[float, float, float] | None): Tuple of 3 float representing actual scan size in mm.
                In the format of Depth x Height x Width. If None and size and spacing are not None,
                physical_size will be inferred.
            spacing (tuple[float, float, float] | None): Tuple of 3 float representing spacing between voxels
                of scan in mm/voxel. In the format of Depth x Height x Width. If None,
                preprocessed scans and annotations retain their original spacing.
        """
        self.scans_write_dir = scans_write_dir
        self.annotation_write_dir = annotation_write_dir
        self.size = size
        self.spacing = spacing

        if physical_size is None and self.spacing is not None and self.size is not None:
            # calculate physical size
            self.physical_size = (
                self.spacing[0] * self.size[0],
                self.spacing[1] * self.size[1],
                self.spacing[2] * self.size[2],
            )

        # If size, spacing and physical size are passed, ensure they are compatible.
        if size is not None and spacing is not None and physical_size is not None:
            assert spacing[0] * size[0] == physical_size[0]
            assert spacing[1] * size[1] == physical_size[1]
            assert spacing[2] * size[2] == physical_size[2]


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
            annotations_path (Path): The path where the annotation associated with the Case is located.
            settings (PreprocessingSettings): The settings determining how the case is preprocessed.
        """
        self.scan_paths = scan_paths
        self.annotations_path = annotations_path
        self.settings = settings

        self.scans: list[sitk.Image]
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
    def write(self) -> tuple[Sequence[Path], Path]:
        """
        Abstract method to be implemented by children that writes the preprocessed scans and annotation to their
        destination and returns the file paths.

        Returns:
            (tuple[Sequence[Path], Path]): A tuple in which the first entry is a sequence of file paths for the scans
                and the second entry is the file path to the corresponding annotation.

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

        scan_paths filenames are assumed to have the following format: <patient_id>_<study_id>_<modality>.mha
        where modality is a three letter string of ['t2w', 'adc', 'hbv']. **NOTE**: the ordering self.scan_path
        and self.scans must remain consistent.

        annotation_path filename is assumed to have the following format: <patient_id>_<study_id>.nii.gz

        Args:
            scan_paths (Sequence[Path]): The set of paths where the scans associated with the Case are located.
                **NOTE**: self.scans will inherit the ordering of scan_paths and must remain consistently ordered.
            annotations_path (Path): The path where the annotation associated with the Case is located.
            settings (PreprocessingSettings): The settings determining how the case is preprocessed.
        """
        super().__init__(scan_paths, annotations_path, settings)

    def read(self) -> None:
        """Reads in scans and annotation into the corresponding scans and annotation attribute."""
        assert len(self.scan_paths) != 0
        self.scans = [sitk.ReadImage(path) for path in self.scan_paths]
        self.annotation = sitk.ReadImage(self.annotations_path)

    def write(self) -> tuple[Sequence[Path], Path]:
        """
        Writes preprocessed scans and annotations from PICAI dataset to disk and returns the scan file paths and
        annotation file path in a tuple.

        Assumes scan_paths and annotation_path filenames follow the format specified in class constructor.
        **NOTE**: ``self.scans`` will inherit the ordering of scan_paths and must remain consistently ordered.

        Output ``scan_paths`` will be located at: ``scans_write_dir/<patient_id>_<stud_id>_<modality_id>.nii.gz``
        where ``<modality_id>`` is a mapping from modality string to a 4 digit number specified by the mapping
        below.

        Returns:
            (tuple[Sequence[Path], Path]): A tuple in which the first entry is a sequence of file paths for the scans
                and the second entry is the file path to the corresponding annotation.
        """
        modality_suffix_map = {"t2w": "0000", "adc": "0001", "hbv": "0002"}
        preprocessed_scan_paths = []
        for path, scan in zip(self.scan_paths, self.scans):
            scan_filename = Path(os.path.basename(path))
            scan_filename_without_extension = scan_filename.stem
            suffix = modality_suffix_map[scan_filename_without_extension[-3:]]
            preprocessed_scan_filename = scan_filename_without_extension[:-3] + suffix + ".nii.gz"
            preprocessed_scan_path = Path(os.path.join(self.settings.scans_write_dir, preprocessed_scan_filename))
            sitk.WriteImage(scan, preprocessed_scan_path, useCompression=True)
            preprocessed_scan_paths.append(preprocessed_scan_path)

        annotation_filename = os.path.basename(self.annotations_path)
        preprocessed_annotation_path = Path(os.path.join(self.settings.annotation_write_dir, annotation_filename))
        sitk.WriteImage(self.annotation, preprocessed_annotation_path, useCompression=True)

        return preprocessed_scan_paths, preprocessed_annotation_path


class PreprocessingError(Exception):
    """Custom exception to be thrown during a failed attempt at processing a Case."""

    pass


class PreprocessingTransform(ABC):
    """Abstract class that represents a transform to be applied to a case."""

    @abstractmethod
    def __call__(self, case: Case) -> Case:
        """
        Abstract method to be implemented by children that applies a transformation
        to the input Case and returns the resulting Case.

        Args:
            case (Case): The case to be processed.

        Returns:
            (Case): The Case after the transformation has been applied.
        """
        raise NotImplementedError


class ResampleToFirstScan(PreprocessingTransform):
    def __call__(self, case: Case) -> Case:
        """
        Resample scans and label to the first scan.

        Args:
            case (Case): The case to be processed.

        Returns:
            (Case): The resampled Case.
        """
        # set up resampler to resolution, field of view, etc. of first scan
        resampler = sitk.ResampleImageFilter()  # default linear
        resampler.SetReferenceImage(case.scans[0])
        resampler.SetInterpolator(sitk.sitkBSpline)

        # resample other images
        case.scans[1:] = [resampler.Execute(scan) for scan in case.scans[1:]]

        # resample annotation
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        case.annotation = resampler.Execute(case.annotation)

        return case


class ResampleSpacing(PreprocessingTransform):
    def __call__(self, case: Case) -> Case:
        """
        Resamples scan to target resolution spacing.

        Args:
            case (Case): The case to be processed.

        Returns:
            (Case): The Case where scans and annotation have specified spacing.
        """
        assert case.settings.spacing is not None
        # resample scans to target resolution
        case.scans = [resample_img(scan, case.settings.spacing, is_label=False) for scan in case.scans]

        # resample annotation to target resolution
        case.annotation = resample_img(case.annotation, case.settings.spacing, is_label=True)

        return case


class CentreCropAndOrPad(PreprocessingTransform):
    def __call__(self, case: Case) -> Case:
        """
        Centre crop and/or pad scans and label.

        Args:
            case (Case): The case to be processed.

        Returns:
            (Case): The Case after centre crop and/or padding.
        """
        assert case.settings.size is not None
        case.annotation = crop_or_pad(case.annotation, case.settings.size, case.settings.physical_size)
        case.scans = [
            crop_or_pad(
                scan,
                case.settings.size,
                case.settings.physical_size,
            )
            for scan in case.scans
        ]

        return case


class AlignOriginAndDirection(PreprocessingTransform):
    def __call__(self, case: Case) -> Case:
        """
        Align the origin and direction of each scan, and label.

        Args:
            case (Case): The case to be processed.

        Returns:
            (Case): The Case after aligning origin and direction.
        """
        case_origin, case_direction, case_spacing = None, None, None
        for scan in case.scans:
            # copy metadata of first scan (nnUNet and nnDetection require this to match exactly)
            if case_origin is None:
                case_origin = scan.GetOrigin()
                case_direction = scan.GetDirection()
                case_spacing = scan.GetSpacing()
            else:
                # check if current scan's metadata is almost equal to the first scan
                assert case_direction is not None
                assert case_spacing is not None
                np.testing.assert_allclose(scan.GetOrigin(), case_origin)
                np.testing.assert_allclose(scan.GetDirection(), case_direction)
                np.testing.assert_allclose(scan.GetSpacing(), case_spacing)

                # copy over first scan's metadata to current scan
                scan.SetOrigin(case_origin)
                scan.SetDirection(case_direction)
                scan.SetSpacing(case_spacing)

        assert all([case_origin, case_direction, case_spacing])
        case.annotation.SetOrigin(case_origin)
        case.annotation.SetDirection(case_direction)
        case.annotation.SetSpacing(case_spacing)

        return case


class BinarizeAnnotation(PreprocessingTransform):
    def __call__(self, case: Case) -> Case:
        """
        Binarize the granular ISUP ≥ 2 annotations.
        Annotations currently exist on 1 - 5 scale based on severity of csPCa.
        This maps annotations to binary using 2 (1 below because of 0-indexing) as the threshold.

        Args:
            case (Case): The case to be processed.

        Returns:
            (Case): The Case after binarizing the annotation.
        """
        annotation_array = sitk.GetArrayFromImage(case.annotation)

        # convert granular PI-CAI csPCa annotation to binary csPCa annotation
        annotation_array = (annotation_array >= 1).astype("uint8")

        # convert label back to SimpleITK
        new_annotation = sitk.GetImageFromArray(annotation_array)
        new_annotation.CopyInformation(case.annotation)
        case.annotation = new_annotation
        return case


def apply_transform(case: Case, transforms: Sequence[PreprocessingTransform]) -> tuple[Sequence[Path], Path]:
    """
    Reads in scans and annotation, applies sequence of transformations, and writes resulting case to disk.
    Returns tuple with scan paths and corresponding annotation path.

    Args:
        case (Case): The case to be processed.
        transforms (Sequence[PreprocessingTransform]): The sequence of transformation to be applied.

    Returns:
        (tuple[Sequence[Path], Path]): A tuple in which the first entry is a sequence of file paths
            for the scans and the second entry is the file path to the corresponding annotation.

    Raises:
        PreprocessingException if an error occurs during preprocessing.
    """
    try:
        case.read()
        processed_case = reduce(lambda acc, f: f(acc), transforms, case)
        return processed_case.write()
    except Exception as e:
        error_path_string = ", ".join([str(path) for path in case.scan_paths] + [str(case.annotations_path)])
        raise PreprocessingError(f"Error preprocessing case with following paths: {error_path_string}") from e


def preprocess(
    cases: list[Case], transforms: Sequence[PreprocessingTransform], num_threads: int = 4
) -> Sequence[tuple[Sequence[Path], Path]]:
    """
    Preprocesses a list of cases according to the specified transformations.

    Args:
        cases (list[Case]): A list of cases to be preprocessed.
        transforms (Sequence[PreprocessingTransform]): The sequence of transformation to be applied.
        num_threads (int): The number of threads to use for preprocessing.

    Returns:
        (Sequence[tuple[Sequence[Path], Path]]): A sequence of tuples in which the first entry is a sequence of
            file paths for the scans and the second entry is the file path to the corresponding annotation.

    Raises:
        PreprocessingException if an error occurs during preprocessing of any of the cases.
    """
    f = partial(apply_transform, transforms=transforms)
    if num_threads >= 2:
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            return list(executor.map(f, cases))
    else:
        return list(map(f, cases))
