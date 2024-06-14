"""Finetunes a pretrained nnUNet model on a new dataset

Assumptions:
- The new dataset must be correctly formatted for nnUNet and present in
    the nnUnet_raw folder (see nnUNet installation instructions and environment variables)
- The output of nnUNetv2_extract_fingerprint and nnUNetv2_plan_experiment
    for the pretraining dataset must be provided. Namely the dataset_fingerprint.json and nnUNetPlans.json
- The new dataset must have the same dimensions for the input and output
    images/volumes/masks as the dataset used for pretraining (ideally the
    same spacings as well as I assume this affects the model?)

"""

import argparse
import os
import shutil
from os import path
from pathlib import Path

import nnunetv2.experiment_planning
import nnunetv2.experiment_planning.plan_and_preprocess_api
import nnunetv2.paths
import nnunetv2.run
import nnunetv2.run.run_training
from nnunetv2.utilities.dataset_name_id_conversion import convert_id_to_dataset_name


def transfer_metadata(
    ft_dataset_id: int, pt_fingerprint_path: Path, pt_plans_path: Path, pt_checkpoint_path: Path
) -> None:
    """Transfers over the fingerprint and plans from the pretrained model.

    Args:
        ft_dataset_id (int): ID for the new finetuning dataset that is
            being used during training
        pt_fingerprint_path (str): Path to the json fingerprint for the
            dataset used during pretraining
        pt_plans_path (str): Path to the plans json for the dataset used
            during pretraining
        pt_checkpoint_path (str): Path to the model checkpoint of the
            pretrained model
    """

    dataset_name = convert_id_to_dataset_name(ft_dataset_id)  # Get name of dataset
    dataset_pp_path = path.join(
        nnunetv2.paths.nnUNet_preprocessed, dataset_name
    )  # Path to folder for preprocessed data
    if not path.exists(dataset_pp_path):
        os.mkdir(dataset_pp_path)

    # Copy over plans, fingerprint, checkpoint and dataset json
    shutil.copy(pt_fingerprint_path, path.join(dataset_pp_path, "dataset_fingerprint.json"))
    shutil.copy(pt_plans_path, path.join(dataset_pp_path, "nnUNetPlans.json"))
    shutil.copy(pt_checkpoint_path, path.join(dataset_pp_path, "pretrained_weights.pth"))


def main() -> None:
    """Parses arguments and passes them to transfer, preprocessing and training functions"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", type=int, required=True, help="The nnUNet dataset ID integer for the fine-tuning dataset"
    )
    parser.add_argument(
        "--pt_d",
        type=int,
        required=False,
        help="The dataset ID integer of the pretraining dataset."
        "Can be omitted if pt_fingerprint and pt_plans flags are both included",
    )
    parser.add_argument(
        "--pt_fingerprint",
        type=str,
        required=False,
        help="Path to the dataset_fingerprint json for the dataset used in pretraining",
    )
    parser.add_argument(
        "--pt_plans", type=str, required=False, help="Path to the experiment plans used for the pretrained model"
    )
    parser.add_argument(
        "--pt_checkpoint",
        type=str,
        required=True,
        help="Path to the model checkpoint for the pretrained model to use as a starting point",
    )
    parser.add_argument(
        "-c", type=str, required=True, help="Configuration that should be trained. Eg. '2d', '3d_fullres'"
    )
    parser.add_argument(
        "-f", type=int, required=True, help="Fold of the 5-fold cross-validation. Should be an inte between 0 and 4"
    )
    parser.add_argument(
        "-tr",
        type=str,
        required=False,
        default="nnUNetTrainer",
        help="Use this flag to specify a custom trainer."
        "Some alternatives are already provided by nnUNet."
        "Eg. 'nnUNetTrainer_5epochs'",
    )
    parser.add_argument(
        "--skip_preprocessing",
        type=bool,
        required=False,
        default=False,
        help="Whether or not to skip the preprocessing step."
        "Only set this to True if you have already run this command with"
        "preprocessing and only want to redo the fine-tuning",
    )

    args = parser.parse_args()

    # Get fingerprint
    if args.pt_fingerprint is None:
        assert args.pt_d is not None, "Either pt_d or pt_fingerprint needs to be provided. Missing both"
        pt_dataset_name = convert_id_to_dataset_name(args.pt_d)
        args.pt_fingerprint = path.join(
            nnunetv2.paths.nnUNet_preprocessed, pt_dataset_name, "dataset_fingerprint.json"
        )

    if args.pt_plans is None:
        assert args.pt_d is not None, "Either pt_d or pt_plans needs to be provided. Missing both"
        pt_dataset_name = convert_id_to_dataset_name(args.pt_d)
        args.pt_plans = path.join(nnunetv2.paths.nnUNet_preprocessed, pt_dataset_name, "nnUNetPlans.json")

    transfer_metadata(
        ft_dataset_id=args.d,
        pt_fingerprint_path=args.pt_fingerprint,
        pt_plans_path=args.pt_plans,
        pt_checkpoint_path=args.pt_checkpoint,
    )

    # Note can possibly extend this to copy dataset json of pretrained
    # dataset and just change number of samples to create new dataset
    # json for finetuning set. For now assume it exists in raw data folder
    ft_dataset_name = convert_id_to_dataset_name(args.d)
    shutil.copy(
        path.join(nnunetv2.paths.nnUNet_raw, ft_dataset_name, "dataset.json"),
        path.join(nnunetv2.paths.nnUNet_preprocessed, ft_dataset_name, "dataset.json"),
    )

    if not args.skip_preprocessing:
        nnunetv2.experiment_planning.plan_and_preprocess_api.preprocess(dataset_ids=[args.d])

    # Can expand this to pass on kwargs if we end up using it alot and want that functionality.
    nnunetv2.run.run_training.run_training(
        dataset_name_or_id=str(args.d), configuration=args.c, fold=args.f, trainer_class_name=args.tr
    )


if __name__ == "__main__":
    main()
