import argparse
import os
from pathlib import Path
from typing import Optional, Sequence

from preprocessing import (
    AlignOriginAndDirection,
    CentreCropAndOrPad,
    MriExam,
    PicaiPreprocessingSettings,
    Resample,
    ResampleToFirstScan,
    preprocess,
)

DEFAULT_TRANSFORMS = [Resample(), CentreCropAndOrPad(), ResampleToFirstScan(), AlignOriginAndDirection()]
DEFAULT_MODALITY_SUFFIXES = ["adc", "t2w", "hbv"]


def preprare_data(
    scans_read_dir: Path,
    annotation_read_dir: Path,
    scans_write_dir: Path,
    annotation_write_dir: Path,
    size: Optional[Sequence[int]] = None,
    physical_size: Optional[Sequence[int]] = None,
    spacing: Optional[Sequence[int]] = None,
    scan_extension: str = "mha",
    annotation_extension: str = ".nii.gz",
    num_threads: int = 4,
) -> None:
    settings = PicaiPreprocessingSettings(
        scans_write_dir,
        annotation_write_dir,
        size,
        physical_size,
        spacing,
    )
    valid_suffixes = tuple([f"{suffix}.{scan_extension}" for suffix in DEFAULT_MODALITY_SUFFIXES])
    valid_annotation_filenames = [f for f in os.listdir(annotation_read_dir) if f.endswith(annotation_extension)]

    samples = []
    for annotation_filename in valid_annotation_filenames:
        annotation_path = Path(os.path.join(annotation_read_dir, annotation_filename))
        annotation_base_filename = annotation_filename.split(".")[0]
        patient_id, study_id = annotation_base_filename.split("_")
        scans_path = os.path.join(scans_read_dir, patient_id)
        scan_filenames = [
            path for path in sorted(os.listdir(scans_path)) if path.endswith(valid_suffixes) and study_id in path
        ]
        scan_paths = [Path(os.path.join(scans_path, f)) for f in scan_filenames]
        sample = MriExam(scan_paths, annotation_path, settings)
        samples.append(sample)

    preprocess(samples, DEFAULT_TRANSFORMS, num_threads=num_threads)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scans_read_dir", type=str, required=True, help="Directory to read scans from.")
    parser.add_argument("--annotation_read_dir", type=str, required=True, help="Directory to read annotation from.")
    parser.add_argument("--scans_write_dir", type=str, required=True, help="Directory to write scans to.")
    parser.add_argument("--annotation_write_dir", type=str, required=True, help="Directory to write annotation to.")
    parser.add_argument(
        "--size",
        type=int,
        nargs="+",
        required=False,
        help="Size to convert images and annotations to. Default is to keep as is.",
    )
    parser.add_argument(
        "--spacing",
        type=float,
        nargs="+",
        required=False,
        help="Spacing to convert images and annotations to. Default is to keep as is.",
    )
    parser.add_argument(
        "--physical_size",
        type=float,
        nargs="+",
        required=False,
        help="Size to convert images and annotations to. Default is to keep as is.",
    )
    parser.add_argument(
        "--scan_extension", type=str, required=False, default="mha", help="Directory to write scans to."
    )
    parser.add_argument(
        "--annotation_extension", type=str, required=False, default="nii.gz", help="Directory to write annotation to."
    )
    parser.add_argument("--num_threads", type=str, default=4, help="Number of threads to use during preprocessing.")

    args = parser.parse_args()
    preprare_data(
        args.scans_read_dir,
        args.annotation_read_dir,
        args.scans_write_dir,
        args.annotation_write_dir,
        tuple(args.size) if args.size is not None else None,
        tuple(args.physical_size) if args.physical_size is not None else None,
        tuple(args.spacing) if args.spacing is not None else None,
        args.scan_extension,
        args.annotation_extension,
    )


if __name__ == "__main__":
    main()
