import os
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from research.picai.data.preprocessing import PicaiCase, PreprocessingSettings


def test_read_and_write_picai_case(tmp_path: Path) -> None:
    x, y, z = 8, 16, 32
    os.mkdir(f"{tmp_path}/read")
    os.mkdir(f"{tmp_path}/write")

    t2w = np.zeros((x, y, z), dtype=np.float32)
    adc = np.ones((x, y, z), dtype=np.float32)
    hbv = np.ones((x, y, z), dtype=np.float32) * 2.0
    annotation = np.ones((x, y, z), dtype=np.float32) * 3.0

    t2w_sitk = sitk.GetImageFromArray(t2w)
    adc_sitk = sitk.GetImageFromArray(adc)
    hbv_sitk = sitk.GetImageFromArray(hbv)
    annotation_sitk = sitk.GetImageFromArray(annotation)

    t2w_filename = Path(f"{tmp_path}/read/file_t2w.mha")
    adc_filename = Path(f"{tmp_path}/read/file_adc.mha")
    hbv_filename = Path(f"{tmp_path}/read/file_hbv.mha")
    annotation_filename = Path(f"{tmp_path}/read/annotation.nii.gz")

    sitk.WriteImage(t2w_sitk, t2w_filename)
    sitk.WriteImage(adc_sitk, adc_filename)
    sitk.WriteImage(hbv_sitk, hbv_filename)
    sitk.WriteImage(annotation_sitk, annotation_filename)

    scan_paths = [t2w_filename, adc_filename, hbv_filename]

    original_np_scans = [t2w, adc, hbv]

    settings = PreprocessingSettings(Path(f"{tmp_path}/write"), Path(f"{tmp_path}/write"), None, None, None)

    case = PicaiCase(scan_paths, annotation_filename, settings)
    case.read()

    read_np_scans = list(map(sitk.GetArrayFromImage, case.scans))
    assert all((original == new).all() for original, new in zip(original_np_scans, read_np_scans))

    assert (annotation == sitk.GetArrayFromImage(case.annotation)).all()

    case.write()

    write_t2w_filename = f"{tmp_path}/write/file_0000.nii.gz"
    write_adc_filename = f"{tmp_path}/write/file_0001.nii.gz"
    write_hbv_filename = f"{tmp_path}/write/file_0002.nii.gz"
    write_annotation_filename = f"{tmp_path}/write/annotation.nii.gz"

    write_scans = [sitk.ReadImage(f) for f in [write_t2w_filename, write_adc_filename, write_hbv_filename]]
    write_np_scans = list(map(sitk.GetArrayFromImage, write_scans))

    assert all((original == new).all() for original, new in zip(original_np_scans, write_np_scans))

    assert (annotation == sitk.GetArrayFromImage(sitk.ReadImage(write_annotation_filename))).all()
