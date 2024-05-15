import argparse
import json
import os
from pathlib import Path
from typing import Optional, Sequence, Tuple

from preprocessing import (
    AlignOriginAndDirection,
    CentreCropAndOrPad,
    PicaiCase,
    PreprocessingSettings,
    Resample,
    ResampleToFirstScan,
    preprocess,
)

DEFAULT_TRANSFORMS = [Resample(), CentreCropAndOrPad(), ResampleToFirstScan(), AlignOriginAndDirection()]


def generate_dataset_json(
    paths_for_each_sample: Sequence[Tuple[Sequence[Path], Path]], write_dir: Path, splits_path: Optional[Path] = None
) -> None:

    # Since we are storing paths inside a json file, we need to convert to strings to be serializable.
    string_paths_for_each_sample = [(list(map(str, tup[0])), str(tup[1])) for tup in paths_for_each_sample]

    if splits_path is None:
        # If splits_path is None, create a singe dataset overview
        json_dict = {}
        json_dict["image_paths"], json_dict["label_paths"] = zip(*string_paths_for_each_sample)
        write_path = os.path.join(write_dir, "train-fold-all.json")
        with open(write_path, "w") as f:
            json.dump(json_dict, f)
    else:
        # If splits_path is not None, create a dataset overview for each split
        with open(splits_path, "r") as splits_f:
            splits = json.load(splits_f)

        for i, split in enumerate(splits):
            # Add image (scan) and label (annotation) paths that include any subject ids from split
            # For both train and val
            train_json_dict, val_json_dict = {}, {}
            train_paths_for_each_sample = [
                tup
                for tup in string_paths_for_each_sample
                if any([subject_id in str(tup[1]) for subject_id in split["train"]])
            ]
            val_paths_for_each_sample = [
                tup
                for tup in string_paths_for_each_sample
                if any([subject_id in str(tup[1]) for subject_id in split["val"]])
            ]
            train_json_dict["image_paths"], train_json_dict["label_paths"] = zip(*train_paths_for_each_sample)
            val_json_dict["image_paths"], val_json_dict["label_paths"] = zip(*val_paths_for_each_sample)

            # Create path of json files and write to disk
            train_write_path = os.path.join(write_dir, f"train-fold-{i}.json")
            val_write_path = os.path.join(write_dir, f"val-fold-{i}.json")

            with open(train_write_path, "w") as f:
                json.dump(train_json_dict, f)

            with open(val_write_path, "w") as f:
                json.dump(val_json_dict, f)


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
