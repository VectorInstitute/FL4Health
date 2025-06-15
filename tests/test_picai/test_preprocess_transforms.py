from collections.abc import Sequence
from functools import partial
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from research.picai.data.preprocessing import (
    AlignOriginAndDirection,
    BinarizeAnnotation,
    Case,
    CentreCropAndOrPad,
    PreprocessingSettings,
    ResampleSpacing,
    ResampleToFirstScan,
)


mock_preprocessing_settings_partial = partial(
    PreprocessingSettings,
    scans_write_dir=Path(""),
    annotation_write_dir=Path(""),
    physical_size=None,
)


class MockCase(Case):
    def __init__(
        self,
        scan_paths: Sequence[Path],
        annotation_path: Path,
        settings: PreprocessingSettings,
        original_scan_sizes: Sequence[tuple[int, int, int]] = [(256, 256, 20), (512, 512, 40), (128, 128, 10)],
        original_annotation_size: tuple[int, int, int] = (384, 384, 30),
    ) -> None:
        super().__init__(scan_paths, annotation_path, settings)
        self.original_scan_sizes = original_scan_sizes
        self.original_annotation_size = original_annotation_size

    def read(self) -> None:
        np_scans = [np.random.rand(*size) for size in self.original_scan_sizes]
        self.scans = [sitk.GetImageFromArray(scan) for scan in np_scans]
        self.annotation = sitk.GetImageFromArray(np.random.randint(0, 5, self.original_annotation_size))

    def write(self) -> tuple[Sequence[Path], Path]:
        return ([Path("") for _ in range(3)], Path(""))


def test_resample_to_first_scan() -> None:
    mock_settings = mock_preprocessing_settings_partial(size=(256, 256, 20), spacing=(1.0, 1.0, 1.0))
    case = MockCase([Path("") for _ in range(3)], Path(""), mock_settings)
    case.read()

    transform = ResampleToFirstScan()
    new_case = transform(case)

    sizes = [item.GetSize() for item in new_case.scans + [new_case.annotation]]
    spacings = [item.GetSpacing() for item in new_case.scans + [new_case.annotation]]

    assert all(x == (20, 256, 256) for x in sizes)
    assert all(x == (1.0, 1.0, 1.0) for x in spacings)


def test_resample_spacing() -> None:
    mock_settings = mock_preprocessing_settings_partial(size=(256, 256, 20), spacing=(2.0, 2.0, 2.0))
    case = MockCase([Path("") for _ in range(3)], Path(""), mock_settings)
    case.read()

    transform = ResampleSpacing()
    new_case = transform(case)

    spacings = [item.GetSpacing() for item in new_case.scans + [new_case.annotation]]

    assert all(x == (2.0, 2.0, 2.0) for x in spacings)


def test_resample_crop_or_pad() -> None:
    mock_settings = mock_preprocessing_settings_partial(size=(256, 256, 20), spacing=(1.0, 1.0, 1.0))
    case = MockCase([Path("") for _ in range(3)], Path(""), mock_settings)
    case.read()

    transform = CentreCropAndOrPad()
    new_case = transform(case)

    sizes = [item.GetSize() for item in new_case.scans + [new_case.annotation]]
    spacings = [item.GetSpacing() for item in new_case.scans + [new_case.annotation]]

    assert all(x == (20, 256, 256) for x in sizes)
    assert all(x == (1.0, 1.0, 1.0) for x in spacings)


def test_align_origin_and_direction() -> None:
    mock_settings = mock_preprocessing_settings_partial(size=(256, 256, 20), spacing=(1.0, 1.0, 1.0))
    case = MockCase([Path("") for _ in range(3)], Path(""), mock_settings)
    case.read()

    transform = AlignOriginAndDirection()
    new_case = transform(case)

    origins = [item.GetOrigin() for item in new_case.scans + [new_case.annotation]]
    dimensions = [item.GetDimension() for item in new_case.scans + [new_case.annotation]]
    origins_target = origins[0]
    dimensions_target = dimensions[0]

    assert all(x == origins_target for x in origins)
    assert all(x == dimensions_target for x in dimensions)


def test_binarize_annotation() -> None:
    mock_settings = mock_preprocessing_settings_partial(size=None, spacing=None)
    case = MockCase([Path("") for _ in range(3)], Path(""), mock_settings)
    case.read()

    transform = BinarizeAnnotation()
    new_case = transform(case)

    annotation_array = sitk.GetArrayFromImage(new_case.annotation)
    unique_values = list(np.unique(annotation_array))

    for val in unique_values:
        assert val in [0, 1]
