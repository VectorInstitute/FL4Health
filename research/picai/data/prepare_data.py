import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from research.picai.data.preprocessing import (
    AlignOriginAndDirection,
    BinarizeAnnotation,
    Case,
    CentreCropAndOrPad,
    PicaiCase,
    PreprocessingSettings,
    ResampleSpacing,
    ResampleToFirstScan,
    preprocess,
)


DEFAULT_TRANSFORMS = [
    ResampleSpacing(),
    CentreCropAndOrPad(),
    ResampleToFirstScan(),
    AlignOriginAndDirection(),
    BinarizeAnnotation(),
]


def get_labels(paths_for_each_sample: Sequence[tuple[Sequence[Path], Path]]) -> Sequence[float]:
    """
    Get the label of each sample. The label is negative if no foreground objects exist, as per annotation,
    else positive.

    Args:
        paths_for_each_sample (Sequence[tuple[Sequence[Path], Path]]): A sequence in which each member
            is tuple where the first entry is a list of scan paths and the second in the annotation path.

    Returns:
        (Sequence[float]): The label for each of the samples.
    """
    _, annotation_paths = zip(*paths_for_each_sample)
    case_labels = []
    for path in annotation_paths:
        annotation = sitk.ReadImage(path)
        annotation_array = sitk.GetArrayFromImage(annotation)
        # Check if the sample contains any foreground objects (ie Is this a positive or negative sample?)
        case_label = float(np.max(annotation_array))
        case_labels.append(case_label)
    return case_labels


def filter_split_on_subject_id(
    scan_annotation_label_list: Sequence[tuple[Sequence[str], str, float]],
    split: dict[str, Sequence[str]],
    train: bool,
) -> dict[str, Sequence[float] | Sequence[str]]:
    """
    Filters the scan_annotation_label_list to only include samples with a subject_id apart of split. Returns Dict with
    image paths, label paths and case labels.

    Args:
        scan_annotation_label_list (Sequence[tuple[list[str], str, float]]): A sequence where each member
            is a tuple where the first entry is a list of scan paths, the second entry is the annotation
            path and the third entry is the label of the sample.
        split (dict[str, Sequence[str]]): A Dict of sequences of subject_ids included in the current split.
            Dict contains two keys: train and val.
        train (bool): Whether to use the train or the test split.

    Returns:
        (dict[str, Sequence[float] | Sequence[str]]): A Dict containing image_paths, label_paths
            and case_label for each sample part of the split.
    """
    train_or_val_string = "train" if train else "val"
    filtered_scan_annotation_label_list: Sequence[tuple[Sequence[str], str, float]] = [
        (scan_paths, annotation_path, label)
        for (scan_paths, annotation_path, label) in scan_annotation_label_list
        if any(subject_id in annotation_path for subject_id in split[train_or_val_string])
    ]
    labeled_data: dict[str, Sequence[float] | Sequence[str]] = {}
    labeled_data["image_paths"], labeled_data["label_paths"], labeled_data["case_label"] = zip(
        *filtered_scan_annotation_label_list
    )
    return labeled_data


def generate_dataset_json(
    paths_for_each_sample: Sequence[tuple[Sequence[Path], Path]],
    write_dir: Path,
    splits_path: Path | None = None,
) -> None:
    """
    Generates JSON file(s) that include the image_paths, label_paths and case_labels.
    If splits_path is supplied, a JSON file will be created for each of the splits.
    If no splits_path is supplied, a single JSON file will be created with all of the samples.

    Args:
        paths_for_each_sample (Sequence[tuple[Sequence[Path], Path]]): A sequence in which each member
            is tuple where the first entry is a list of scan paths and the second in the annotation path.
        write_dir (Path): The directory to write the dataset file(s).
        splits_path (Path | None): The path to the desired spits. JSON file with key for each split.
            Each key contains nested keys train and val. Inside the nested keys are lists of subject_id
            strings to be included in the split.
    """
    labels = get_labels(paths_for_each_sample)

    # Since we are storing paths inside a json file, we need to convert to strings to be serializable.
    # Also pack in label for each samples in the tuple containing scan paths and annotation path.
    scan_annotation_label_list = [
        (list(map(str, scan_paths)), str(annotation_path), label)
        for (scan_paths, annotation_path), label in zip(paths_for_each_sample, labels)
    ]

    if splits_path is None:
        # If splits_path is None, create a singe dataset overview
        labeled_data: dict[str, Sequence[str] | Sequence[float]] = {}
        labeled_data["image_paths"], labeled_data["label_paths"], labeled_data["case_label"] = zip(
            *scan_annotation_label_list
        )
        write_path = os.path.join(write_dir, "train-fold-all.json")
        with open(write_path, "w") as f:
            json.dump(labeled_data, f)
    else:
        # If splits_path is not None, create a dataset overview for each split
        with open(splits_path, "r") as splits_f:
            splits = json.load(splits_f)

        for i, split in enumerate(splits):
            # Add image (scan) and label (annotation) paths that include any subject ids from split
            # For both train and val
            train_labeled_data = filter_split_on_subject_id(scan_annotation_label_list, split, train=True)
            val_labeled_data = filter_split_on_subject_id(scan_annotation_label_list, split, train=False)

            # Create path of json files and write to disk
            train_write_path = os.path.join(write_dir, f"train-fold-{i}.json")
            val_write_path = os.path.join(write_dir, f"val-fold-{i}.json")

            with open(train_write_path, "w") as f:
                json.dump(train_labeled_data, f)

            with open(val_write_path, "w") as f:
                json.dump(val_labeled_data, f)


def prepare_data(
    scans_read_dir: Path,
    annotation_read_dir: Path,
    scans_write_dir: Path,
    annotation_write_dir: Path,
    overview_write_dir: Path,
    size: tuple[int, int, int] | None = None,
    physical_size: tuple[float, float, float] | None = None,
    spacing: tuple[float, float, float] | None = None,
    scan_extension: str = "mha",
    annotation_extension: str = ".nii.gz",
    num_threads: int = 4,
    splits_path: Path | None = None,
) -> None:
    """
    Runs preprocessing on data with specified settings.

    Args:
        scans_read_dir (Path): The path to read the scans from. Should be a directory with subdirectories
            for each patient_id. Inside the subdirectories should be all the scan files for a given patient.
        annotation_read_dir (Path): The path to read the annotations from. Should be a flat directory with all
            annotation files.
        scans_write_dir (Path): The path to write the scans to. All scans are written into same directory.
        annotation_write_dir (Path): The path to write the annotations to. All annotations are written into same
            directory.
        overview_write_dir (Path): The path where the dataset json files are located. For each split 1-5,
            there is a train and validation file with scan paths, label paths and case labels.
        size (tuple[int, int, int] | None, optional): Desired dimensions of preprocessed scans in voxels.
            Triplet of the form: Depth x Height x Width. Defaults to None.
        physical_size (tuple[float, float, float] | None, optional): Desired dimensions of preprocessed scans in mm.
            Simply the product of the number of voxels by the spacing along a particular dimension: Triplet of the
            form: Depth x Height x Width. Defaults to None.
        spacing (tuple[float, float, float] | None, optional): Desired spacing of preprocessed scans in in mm/voxel.
            Triplet of the form: Depth x Height x Width. Defaults to None.
        scan_extension (str, optional): The expected extension of scan file paths. Defaults to "mha".
        annotation_extension (str, optional): The expected extension of annotation file paths. Defaults to ".nii.gz".
        num_threads (int, optional): The number of threads to use during preprocessing. Defaults to 4.
        splits_path (Path | None, optional): The path to the file containing the splits. Defaults to None.
    """
    settings = PreprocessingSettings(
        scans_write_dir,
        annotation_write_dir,
        size,
        physical_size,
        spacing,
    )
    valid_annotation_filenames = [f for f in os.listdir(annotation_read_dir) if f.endswith(annotation_extension)]

    samples: list[Case] = []
    for annotation_filename in valid_annotation_filenames:
        # Annotation filename is subject id (ie patient_id study_id)
        # We use it to get the corresponding scan paths
        annotation_path = Path(os.path.join(annotation_read_dir, annotation_filename))
        # split on filename instead of stem on path since extension can have multiple "." in it.
        annotation_base_filename = annotation_filename.split(".")[0]

        # All of the scans for a given patient, even if there are multiple sets from different study ids,
        # are stored in a single folder under patient id in the raw data.
        patient_id, study_id = annotation_base_filename.split("_")
        scans_path = os.path.join(scans_read_dir, patient_id)

        # Get all the filenames that match the extension we expect and have the proper study id.
        scan_filenames = [
            path for path in os.listdir(scans_path) if study_id in path and path.endswith(scan_extension)
        ]

        # Extract the scans that we care about (t2w, adc, hbv) and store in a list ordered as we expect.
        t2w_scan_filename = [scan_filename for scan_filename in scan_filenames if "t2w" in scan_filename]
        adc_scan_path_filename = [scan_filename for scan_filename in scan_filenames if "adc" in scan_filename]
        hbv_scan_path_filename = [scan_filename for scan_filename in scan_filenames if "hbv" in scan_filename]
        assert len(t2w_scan_filename) == 1 and len(adc_scan_path_filename) == 1 and len(hbv_scan_path_filename) == 1
        ordered_scan_filenames = t2w_scan_filename + adc_scan_path_filename + hbv_scan_path_filename
        scan_paths = [Path(os.path.join(scans_path, f)) for f in ordered_scan_filenames]

        # Create sample and add to list of cases to be processed.
        sample = PicaiCase(scan_paths, annotation_path, settings)
        samples.append(sample)

    # Preprocess list of samples, generate dataset overviews and write them to disk
    paths_for_each_sample = preprocess(samples, DEFAULT_TRANSFORMS, num_threads=num_threads)
    generate_dataset_json(paths_for_each_sample, overview_write_dir, splits_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scans_read_dir", type=str, required=True, help="Directory to read scans from.")
    parser.add_argument("--annotation_read_dir", type=str, required=True, help="Directory to read annotation from.")
    parser.add_argument("--scans_write_dir", type=str, required=True, help="Directory to write scans to.")
    parser.add_argument("--annotation_write_dir", type=str, required=True, help="Directory to write annotation to.")
    parser.add_argument(
        "--overview_write_dir", type=str, required=True, help="Directory to write dataset overviews to."
    )
    parser.add_argument(
        "--size",
        type=int,
        nargs="+",
        required=False,
        help="Size in pixels to convert images and annotations to (Depth x Height x Width). Default is to keep as is.",
    )
    parser.add_argument(
        "--spacing",
        type=float,
        nargs="+",
        required=False,
        help="Spacing in mm/pixel to convert images and annotations to (Depth x Height x Width)."
        "Default is to keep as is.",
    )
    parser.add_argument(
        "--physical_size",
        type=float,
        nargs="+",
        required=False,
        help="Size in number of voxels * spacing to convert images and annotations to (Depth x Height x Width)"
        "Default is to keep as is.",
    )
    parser.add_argument(
        "--scan_extension", type=str, required=False, default="mha", help="The expected extension of scan files."
    )
    parser.add_argument(
        "--annotation_extension",
        type=str,
        required=False,
        default="nii.gz",
        help="The expected extension of annotation files.",
    )
    parser.add_argument("--num_threads", type=int, default=4, help="Number of threads to use during preprocessing.")
    parser.add_argument(
        "--splits_path", type=str, default=None, help="The path to the json file containing the splits."
    )

    args = parser.parse_args()

    if args.size is not None and len(args.size) != 3:
        raise ValueError("Argument size must have length 3")
    size = (int(args.size[0]), int(args.size[1]), int(args.size[2])) if args.size else None

    if args.physical_size is not None and len(args.physical_size) != 3:
        raise ValueError("Argument physical_size must have length 3")
    physical_size = (
        (float(args.physical_size[0]), float(args.physical_size[1]), float(args.physical_size[2]))
        if args.physical_size
        else None
    )

    if args.spacing is not None and len(args.spacing) != 3:
        raise ValueError("Argument spacing must have length 3")
    spacing = (float(args.spacing[0]), float(args.spacing[1]), float(args.spacing[2])) if args.spacing else None

    prepare_data(
        args.scans_read_dir,
        args.annotation_read_dir,
        args.scans_write_dir,
        args.annotation_write_dir,
        args.overview_write_dir,
        size,
        physical_size,
        spacing,
        args.scan_extension,
        args.annotation_extension,
        args.num_threads,
        args.splits_path,
    )


if __name__ == "__main__":
    main()
