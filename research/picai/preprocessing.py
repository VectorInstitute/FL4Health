from __future__ import annotations

import os
from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial, reduce
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import SimpleITK as sitk
from preprocessing_utils import crop_or_pad, resample_img


@dataclass
class PicaiPreprocessingSettings:
    def __init__(
        self,
        scans_write_dir: Path,
        annotation_write_dir: Path,
        size: Optional[Sequence[int]],
        physical_size: Optional[Sequence[float]],
        spacing: Optional[Sequence[float]],
        modality_suffix_map: Dict[str, str] = {"t2w": "001", "adc": "001", "hbv": "002"},
    ) -> None:

        self.scans_write_dir = scans_write_dir
        self.annotation_write_dir = annotation_write_dir
        self.size = size
        self.physical_size = physical_size
        self.spacing = spacing
        self.modality_suffix_map = modality_suffix_map

    def __post_init__(self) -> None:
        if self.physical_size is None and self.spacing is not None and self.size is not None:
            # calculate physical size
            self.physical_size = [
                voxel_spacing * num_voxels for voxel_spacing, num_voxels in zip(self.spacing, self.size)
            ]

        if self.spacing is None and self.physical_size is not None and self.size is not None:
            # calculate spacing
            self.spacing = [size / num_voxels for size, num_voxels in zip(self.physical_size, self.size)]


@dataclass
class MriExam:
    def __init__(
        self,
        scan_paths: Sequence[Path],
        annotations_path: Path,
        settings: PicaiPreprocessingSettings,
        file_extension: str = "mha",
    ):
        self.scan_paths = scan_paths
        self.annotations_path = annotations_path
        self.settings = settings
        self.file_extension = file_extension

        self.scans: List[sitk.Image]
        self.annotation: sitk.Image

    def read(self) -> None:
        assert len(self.scans) != 0
        self.scans = [sitk.ReadImage(path) for path in sorted(self.scan_paths)]
        self.annotation = sitk.ReadImage(self.annotations_path)

    def write(self) -> None:
        scan_paths = [path for path in sorted(self.scan_paths)]
        for path, scan in zip(scan_paths, self.scans):
            filename = os.path.basename(path)
            file = filename.split(".")[0]
            suffix = self.settings.modality_suffix_map[file[:-3]]
            preprocessed_filename = file[:-3] + suffix + ".nii.gz"
            preprocessed_path = os.path.join(self.settings.scans_write_dir, preprocessed_filename)
            sitk.WriteImage(scan, preprocessed_path, useCompression=True)


class PreprocessingTransform:
    @abstractmethod
    def __call__(self, mri: MriExam) -> MriExam:
        raise NotImplementedError


class ResampleToFirstScan(PreprocessingTransform):
    def __call__(self, mri: MriExam) -> MriExam:
        """Resample scans and label to the first scan"""
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
    def __call__(self, mri: MriExam) -> MriExam:
        # resample scans to target resolution
        mri.scans = [resample_img(scan, mri.settings.spacing, is_label=False) for scan in mri.scans]

        # resample annotation to target resolution
        mri.annotation = resample_img(mri.annotation, mri.settings.spacing, is_label=True)

        return mri


class CentreCropAndOrPad(PreprocessingTransform):
    """Centre crop and/or pad scans and label"""

    def __call__(self, mri: MriExam) -> MriExam:
        mri.scans = [
            crop_or_pad(
                scan,
                mri.settings.size,
                mri.settings.physical_size,
            )
            for scan in mri.scans
        ]

        mri.annotation = crop_or_pad(mri.annotation, mri.settings.size, mri.settings.physical_size)

        return mri


class AlignOriginAndDirection(PreprocessingTransform):
    """Align the origin and direction of each scan, and label"""

    def __call__(self, mri: MriExam) -> MriExam:
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


def apply_transform(mri: MriExam, transforms: Sequence[PreprocessingTransform]) -> None:
    mri.read()
    reduce(lambda acc, f: f(acc), transforms, mri)


def preprocess(mris: List[MriExam], transforms: Sequence[PreprocessingTransform], num_threads: int = 4) -> None:
    f = partial(apply_transform, transforms=transforms)
    if num_threads >= 2:
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            executor.map(f, mris)
    else:
        map(f, mris)
