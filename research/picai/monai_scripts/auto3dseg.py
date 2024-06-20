"""Uses Auto3dSeg from monai to train a segmentation model

-Assumes dataset is formatted following nnUNet structure for consistency
"""

import argparse
import os
from os.path import join
from pathlib import Path
from typing import Optional

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import save_json


def gen_dataset_list(data_dir: Path, output_path: Optional[str] = None, ext: str = ".nii.gz") -> None:
    """Generates a MONAI dataset list for an nnUNet structured dataset

    Note: rather than generating image and label keys as is typically done
    in MONAI, this function generates a datalist with channel keys in place
    of the image key. Channels must be concatenated before training using
    a MONAI transform. Channel keys are chXXXX where XXXX is the channel number

    Args:
        data_dir: Path to the nnUNet_raw dataset.
        output_path: Where and what to save the file as. Must be a json.
            Default is to save as datalist.json in the data_dir
    """

    train_dir = join(data_dir, "imagesTr")
    test_dir = join(data_dir, "testTr")

    # Not sure if it is more efficient to figure num channels this way or to just load the dataset.json
    # and get it from there
    channels = np.unique([int(file.split(".")[0][-4:]) for file in os.listdir(train_dir)])

    # Initialize datalist
    datalist = {"testing": [], "training": []}  # type: dict[str, list[dict[str, str]]]

    if os.path.exists(test_dir):  # nnUNet Datasets do not always have test sets
        # Populate test cases
        test_cases = np.unique(
            [file.split(".")[0][:-5] for file in os.listdir(test_dir) if ("._" not in file) and (ext in file)]
        )
        for case in test_cases:
            entry = {}
            for c in channels:
                entry.update({f"ch{c:04d}": "./imagesTs/" + case + f"_{c:04d}" + ext})
            datalist["testing"].append(entry)

    # Populate training cases
    train_cases = np.unique(
        [file.split(".")[0][:-5] for file in os.listdir(train_dir) if ("._" not in file) and (ext in file)]
    )
    for case in train_cases:
        entry = {"label": "./labelsTr/" + case + ext}
        for c in channels:
            entry.update({f"ch{c:04d}": "./imagesTs/" + case + f"_{c:04d}" + ext})
        datalist["training"].append(entry)

    if output_path is None:
        output_path = join(data_dir, "datalist.json")

    save_json(datalist, output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the nnUNet_raw formatted dataset folder")
    parser.add_argument(
        "--ds_list",
        type=str,
        required=False,
        help="[OPTIONAL]Path to the dataset list file for the dataset. If it doesn't exist\
            one is created and saved to this path",
    )
    parser.add_argument(
        "--file_ending",
        type=str,
        required=False,
        default=".nii.gz",
        help="[OPTIONAL]The file ending of the input images and labels. Default is .nii.gz",
    )

    args = parser.parse_args()

    if args.ds_list is None or not os.path.isfile(args.ds_list):
        gen_dataset_list(data_dir=args.data_dir, output_path=args.ds_list, ext=args.file_ending)


if __name__ == "__main__":
    main()
