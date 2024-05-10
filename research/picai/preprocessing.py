import os
from functools import reduce
from pathlib import Path
from typing import Sequence, List
import numpy as np
import SimpleITK as sitk

from dataclasses import dataclass
from abc import abstractmethod

from preprocessing_utils import crop_or_pad, resample_img


@dataclass
class PicaiPreprocessingSettings:
    spacing: Sequence[int]
    size: Sequence[int]
    physical_size: Sequence[int]


@dataclass
class MriExam:
    def __init__(
        self,
        scans_dir: Path,
        annotations_path: Path,
        settings: PicaiPreprocessingSettings,
        file_extension: str = "mha",
    ):
        self.scans_dir = scans_dir
        self.annotations_path = annotations_path
        self.settings = settings
        self.file_extension = file_extension

        self.scans: List[sitk.Image]
        self.annotation: sitk.Image

    def __post_init__(self) -> None:
        self.scans = self.read_scans()
        self.annotation = self.read_annotation()

    def read_scans(self) -> List[sitk.Image]:
        return [sitk.ReadImage(path) for path in os.listdir(self.scans_dir) if path.endswith(self.file_extension)]

    def read_annotation(self) -> sitk.Image:
        return sitk.ReadImage(self.annotations_path)


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
        mri.scans = [
            resample_img(scan, mri.settings.spacing, is_label=False)
            for scan in mri.scans
        ]

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


def preprocess(transformations: Sequence[PreprocessingTransform], mri: MriExam) -> MriExam:
    return reduce(lambda acc, f: f(acc), transformations, mri)
