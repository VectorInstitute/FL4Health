"""
Uses Auto3dSeg from monai to train a segmentation model.

-Assumes dataset is formatted following nnUNet structure for consistency
"""

import argparse
import os
from os.path import join

import numpy as np
import yaml
from batchgenerators.utilities.file_and_folder_operations import load_json, save_json
from monai.apps.auto3dseg.auto_runner import AutoRunner


def gen_dataset_list(data_dir: str, output_path: str | None = None, ext: str = ".nii.gz") -> str:
    """
    Generates a MONAI dataset list for an nnUNet structured dataset.

    **NOTE**: Rather than having a single image and label, this checks for multiple channels following the nnunet
    dataset formatting guidelines, and passes a list of filepaths for each channel as the value for the image key

    Args:
        data_dir (str): Path to the nnUNet_raw dataset.
        output_path (str | None, optional): Where and what to save the file as. Must be a json. Default is to save as
            datalist.json in the data_dir. Defaults to None.
        ext (str, optional): Extension to use. Defaults to ".nii.gz".

    Returns:
        (str): The path to where the datalist file was saved
    """
    train_dir = join(data_dir, "imagesTr")
    test_dir = join(data_dir, "testTr")

    # Get the list of unique channel identifiers
    dataset_json = load_json(join(data_dir, "dataset.json"))
    channels = [int(dataset_json["labels"][ch]) for ch in dataset_json["labels"]]

    # Initialize datalist
    # The values to the testing and training keys should be a list of dictionaries
    # where each dictionary contains information about a single case
    datalist: dict[str, list] = {"testing": [], "training": []}

    # nnUNet datasets store images as unique-case-identifier_xxxx.ext
    # xxxx is a 4 digit integer representing the channel/modality.
    # ext is the file extension
    # Labels are stored as unique-case-identifier.ext as they do not have multiple channels

    if os.path.exists(test_dir):  # nnUNet Datasets do not always have test sets
        # Get test case identifiers
        test_cases = np.unique(
            [file.split(".")[0][:-5] for file in os.listdir(test_dir) if ("._" not in file) and (ext in file)]
        )
        # Create a list of filenames for each channel and store that as the value to the image key
        for case in test_cases:
            case_files = []
            for c in channels:
                case_files.append("./imagesTs/" + case + f"_{c:04d}" + ext)
            datalist["testing"].append({"image": case_files})

    # Get train case identifiers
    train_cases = np.unique(
        [file.split(".")[0][:-5] for file in os.listdir(train_dir) if ("._" not in file) and (ext in file)]
    )
    # Create a list of filenames for each channel and store that as the value to the image key
    for case in train_cases:
        case_files = []
        for c in channels:
            case_files.append("./imagesTr/" + case + f"_{c:04d}" + ext)
        # We also include the path to the label as the value for the label key
        datalist["training"].append({"image": case_files, "label": "./labelsTr/" + case + ext})

    # Save datalist json and return output path
    if output_path is None:
        output_path = join(data_dir, "datalist.json")

    save_json(datalist, output_path)

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True, help="Path to the nnUNet_raw formatted dataset folder")
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Path to the directory where the monai autoseg3d can store intermediate and output files",
    )
    parser.add_argument(
        "--ds-list",
        type=str,
        required=False,
        help="[OPTIONAL] Path to the dataset list file for the dataset. If it doesn't exist\
            one is created and saved to this path",
    )
    parser.add_argument(
        "--file-ending",
        type=str,
        required=False,
        default=".nii.gz",
        help="[OPTIONAL] The file ending of the input images and labels. Default is .nii.gz",
    )
    parser.add_argument(
        "--train-params",
        type=str,
        required=False,
        help="[OPTIONAL] Path to a training parameters yaml file. Can be used to override the default parameters of \
            some of the algorithms in the autoseg3d pipeline. There is not extensive list of all the training \
            parameters as they vary for each algorithm, but they can be found in the train.py script generated for \
            each of the models in the autoseg3d ensemble. The parameters to control training length are num_epochs \
            and num_epochs_per_validation. You can make things faster by changing num_warmup_epochs to 1. \
            If you are using a dummy dataset with only a few samples, you might need to override num_images_per_batch",
    )
    parser.add_argument(
        "--num-folds",
        type=int,
        required=False,
        default=5,
        help="[OPTIONAL] The number of cross validation folds for all algos in autoseg3d. Default is 5",
    )

    args = parser.parse_args()

    # Generate dataset list file
    if args.ds_list is None or not os.path.isfile(args.ds_list):
        args.ds_list = gen_dataset_list(data_dir=args.data_dir, output_path=args.ds_list, ext=args.file_ending)

    # Initialize Autoseg3d autorunner
    runner = AutoRunner(
        work_dir=args.output_dir, input={"modality": "MRI", "datalist": args.ds_list, "dataroot": args.data_dir}
    )

    # Modify training parameters
    if args.train_params is not None:
        with open(args.train_params, "r") as file:
            train_params = yaml.safe_load(file)
    else:
        train_params = {}

    runner.set_training_params(train_params)
    runner.set_num_fold(num_fold=args.num_folds)

    # Run autosegmentation pipeline
    runner.run()


if __name__ == "__main__":
    main()
