import argparse
import json
import os
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import SimpleITK as sitk
from preprocessing import (
    AlignOriginAndDirection,
    BinarizeAnnotation,
    CentreCropAndOrPad,
    PicaiCase,
    PreprocessingSettings,
    Resample,
    ResampleToFirstScan,
    preprocess,
)

DEFAULT_TRANSFORMS = [
    Resample(),
    CentreCropAndOrPad(),
    ResampleToFirstScan(),
    AlignOriginAndDirection(),
    BinarizeAnnotation(),
]


def get_labels(paths_for_each_sample: Sequence[Tuple[Sequence[Path], Path]]) -> Sequence[float]:
    _, annotation_paths = zip(*paths_for_each_sample)
    case_labels = []
    for path in annotation_paths:
        annotation = sitk.ReadImage(path)
        annotation_array = sitk.GetArrayFromImage(annotation)
        case_label = float(np.max(annotation_array))
        case_labels.append(case_label)
    return case_labels


def generate_dataset_json(
    paths_for_each_sample: Sequence[Tuple[Sequence[Path], Path]],
    write_dir: Path,
    splits_path: Optional[Path] = None,
) -> None:

    labels = get_labels(paths_for_each_sample)

    # Since we are storing paths inside a json file, we need to convert to strings to be serializable.
    # Also pack in label for each samples in the tuple containing scan paths and annotation path.
    scan_annotation_label_list = [
        (list(map(str, tup[0])), str(tup[1]), lbl) for tup, lbl in zip(paths_for_each_sample, labels)
    ]

    if splits_path is None:
        # If splits_path is None, create a singe dataset overview
        d = {}
        d["image_paths"], d["label_paths"], d["case_label"] = zip(*scan_annotation_label_list)
        write_path = os.path.join(write_dir, "train-fold-all.json")
        with open(write_path, "w") as f:
            json.dump(d, f)
    else:
        # If splits_path is not None, create a dataset overview for each split
        with open(splits_path, "r") as splits_f:
            splits = json.load(splits_f)

        for i, split in enumerate(splits):
            # Add image (scan) and label (annotation) paths that include any subject ids from split
            # For both train and val
            train_d, val_d = {}, {}
            train_list = [
                tup
                for tup in scan_annotation_label_list
                if any([subject_id in str(tup[1]) for subject_id in split["train"]])
            ]
            val_list = [
                tup
                for tup in scan_annotation_label_list
                if any([subject_id in str(tup[1]) for subject_id in split["val"]])
            ]
            train_d["image_paths"], train_d["label_paths"], train_d["case_label"] = zip(*train_list)
            val_d["image_paths"], val_d["label_paths"], val_d["case_label"] = zip(*val_list)

            # Create path of json files and write to disk
            train_write_path = os.path.join(write_dir, f"train-fold-{i}.json")
            val_write_path = os.path.join(write_dir, f"val-fold-{i}.json")

            with open(train_write_path, "w") as f:
                json.dump(train_d, f)

            with open(val_write_path, "w") as f:
                json.dump(val_d, f)


def preprare_data(
    scans_read_dir: Path,
    annotation_read_dir: Path,
    scans_write_dir: Path,
    annotation_write_dir: Path,
    overview_write_dir: Path,
    size: Optional[Sequence[int]] = None,
    physical_size: Optional[Sequence[int]] = None,
    spacing: Optional[Sequence[int]] = None,
    scan_extension: str = "mha",
    annotation_extension: str = ".nii.gz",
    num_threads: int = 4,
    splits_path: Optional[Path] = None,
) -> None:
    settings = PreprocessingSettings(
        scans_write_dir,
        annotation_write_dir,
        size,
        physical_size,
        spacing,
    )
    valid_annotation_filenames = [f for f in os.listdir(annotation_read_dir) if f.endswith(annotation_extension)]

    samples = []
    for annotation_filename in valid_annotation_filenames:
        # Annotation filename is subject id (ie patient_id study id)
        # We use it to get the corresponding scan paths
        annotation_path = Path(os.path.join(annotation_read_dir, annotation_filename))
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
        t2w_scan_filename = scan_filenames[[i for i, sf in enumerate(scan_filenames) if "t2w" in sf][0]]
        adc_scan_path_filename = scan_filenames[[i for i, sf in enumerate(scan_filenames) if "adc" in sf][0]]
        hbv_scan_path_filename = scan_filenames[[i for i, sf in enumerate(scan_filenames) if "hbv" in sf][0]]
        ordered_scan_filenames = [t2w_scan_filename, adc_scan_path_filename, hbv_scan_path_filename]
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
    parser.add_argument("--num_threads", type=int, default=4, help="Number of threads to use during preprocessing.")
    parser.add_argument(
        "--splits_path", type=str, default=None, help="The path to the json file containing the splits."
    )

    args = parser.parse_args()
    preprare_data(
        args.scans_read_dir,
        args.annotation_read_dir,
        args.scans_write_dir,
        args.annotation_write_dir,
        args.overview_write_dir,
        tuple(args.size) if args.size is not None else None,
        tuple(args.physical_size) if args.physical_size is not None else None,
        tuple(args.spacing) if args.spacing is not None else None,
        args.scan_extension,
        args.annotation_extension,
        args.num_threads,
        args.splits_path,
    )


if __name__ == "__main__":
    main()
