"""Automates simple transfer learning for nnUNetv2."""

import argparse
import multiprocessing
import os
import shutil
from multiprocessing import Process

import torch
from batchgenerators.utilities.file_and_folder_operations import load_json, save_json
from nnunetv2.experiment_planning.plan_and_preprocess_api import extract_fingerprints, preprocess_dataset
from nnunetv2.paths import nnUNet_preprocessed, nnUNet_raw
from nnunetv2.run.run_training import run_training
from nnunetv2.utilities.dataset_name_id_conversion import convert_id_to_dataset_name


def setup_device(device: str) -> torch.device:
    """
    Checks the device being used, sets the number of threads and returns a pytorch device.


    Args:
        device (str): Either 'cpu', 'cuda' or 'mps'

    Returns:
        (torch.device): The corresponding pytorch.device object
    """
    assert device in [
        "cpu",
        "cuda",
        "mps",
    ], f"-device must be either cpu, mps or cuda. Other devices are not tested/supported. Got: {device}."
    if device == "cpu":
        # let's allow torch to use hella threads
        torch.set_num_threads(multiprocessing.cpu_count())
        return torch.device("cpu")
    if device == "cuda":
        # multithreading in torch doesn't help nnU-Net if run on GPU
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        return torch.device("cuda")
    return torch.device("mps")


def move_plans_between_datasets(finetune_id: int, pretrain_plans: str, finetune_plans_identifier: str) -> None:
    """
    Transfers the necessary files needed from pretraining to the relevant finetuning directories. Makes changes to
    the plans file to ensure finetuning runs without issue.

    Args:
        finetune_id (int): The nnunet dataset id for the finetuning dataset
        pretrain_plans (str): Path to the nnunet plans json used for
            pretraining
        finetune_plans_identifier (str): What to name the plans for
            finetuning (no extension)
    """
    ft_dataset_name = convert_id_to_dataset_name(finetune_id)
    plans = load_json(pretrain_plans)  # Will convert plans into ft_plans

    # Change the dataset name
    plans["dataset_name"] = ft_dataset_name

    # Define some variables needed for making further changes to plans file
    pt_plans_identifier = os.path.basename(pretrain_plans).split(".")[0]
    num_tr = load_json(os.path.join(nnUNet_raw, ft_dataset_name, "dataset.json"))[
        "numTraining"
    ]  # Number of training samples
    bs_5percent = round(num_tr * 0.05)

    # Edit parameters for each configuration
    for c in plans["configurations"]:
        # Change the data identifier
        if "data_identifier" in plans["configurations"][c]:
            old_data_identifier = plans["configurations"][c]["data_identifier"]
            if old_data_identifier.startswith(pt_plans_identifier):
                # The data identifier is typically a combination of the plans identifier and the config
                new_data_identifier = finetune_plans_identifier + old_data_identifier[len(pt_plans_identifier) :]
            else:
                new_data_identifier = finetune_plans_identifier + "_" + old_data_identifier
            plans["configurations"][c]["data_identifier"] = new_data_identifier

        # Ensure the batch size is within nnUNet limits
        if "batch_size" in plans["configurations"][c]:
            old_bs = plans["configurations"][c]["batch_size"]
            new_bs = max(min(old_bs, bs_5percent), 2)
            plans["configurations"][c]["batch_size"] = new_bs

    ft_plans_path = os.path.join(nnUNet_preprocessed, ft_dataset_name, finetune_plans_identifier + ".json")
    save_json(plans, ft_plans_path, sort_keys=False)


def transfer_metadata(finetune_id: int, pretrain_id: int, pretrain_plans: str) -> str:
    """
    Transfers and or modifies metadata necessary for transfer learning.

    Ensures the dataset json, dataset fingerprint and nnunet plans are all ready for transfer learning. Some necessary
    changes are made to the plans from pretraining to allow finetuning to occur without issue. Note that this
    function saves the plans for finetuning as 'pt_plans_identifier-ptABC' where ABC is pretrain_id. If a plan file
    is already found that matches this pattern, the function does not overwrite it.


    Args:
        finetune_id (int): The nnunet id of the finetuning dataset
        pretrain_id (int): The nnunet id of the pretraining dataset
        pretrain_plans (str): Path to the nnunet plans json used for pretraining

    Returns:
        (str): The name of the plans file for finetuning (without an extension)
    """
    # Extract Fingerprint if needed
    ft_dataset_name = convert_id_to_dataset_name(finetune_id)
    fingerprint_path = os.path.join(nnUNet_preprocessed, ft_dataset_name, "dataset_fingerprint.json")
    if not os.path.isfile(fingerprint_path):
        extract_fingerprints(dataset_ids=[finetune_id])
    else:
        print("INFO: The finetuning dataset fingerprint has already been extracted. Skipping fingerprint extraction")

    # Transfer pretrained model plans if needed
    pt_plans_identifier = os.path.basename(pretrain_plans).split(".")[0]
    ft_plans_identifier = pt_plans_identifier + "-pt%03.0d" % pretrain_id
    ft_plans_path = os.path.join(nnUNet_preprocessed, ft_dataset_name, ft_plans_identifier + ".json")
    if not os.path.isfile(ft_plans_path):
        move_plans_between_datasets(
            finetune_id=finetune_id, pretrain_plans=pretrain_plans, finetune_plans_identifier=ft_plans_identifier
        )
    else:
        print("INFO: The {ft_plans_identifier} plans file already exists at {ft_plans_path}. Using that instead")

    # Copy dataset json if needed
    if not os.path.isfile(os.path.join(nnUNet_preprocessed, ft_dataset_name, "dataset.json")):
        shutil.copy(
            os.path.join(nnUNet_raw, ft_dataset_name, "dataset.json"),
            os.path.join(nnUNet_preprocessed, ft_dataset_name, "dataset.json"),
        )

    return ft_plans_identifier


def check_configs(plans_path: str, configs: list) -> None:
    """
    Raises an error if configs are not found in plans json.

    Args:
        plans_path (str): Path to the plans file to check.
        configs (list): A list of nnunet configs for which to check.
    """
    plans = load_json(plans_path)
    for c in configs:
        assert c in plans["configurations"], (
            f"Did not find {c} config in {plans_path}. Note that 3d_lowres and 3d_cascade_fullres are not created for "
            "datasets with small image sizes"
        )


def main() -> None:
    """Parses arguments and opens subprocesses to train specified models."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--finetune_id",
        type=int,
        required=True,
        help="The nnUNet dataset ID \
            integer for the fine-tuning dataset",
    )
    parser.add_argument(
        "--pretrain_id",
        type=int,
        required=True,
        help="The nnunet dataset ID integer of the pretraining dataset. \
            if the pretrain plans are given using the --pretrain_plans flag, \
            then pretrain id is only used for filenaming purposes and can be set \
            to any arbitrary id so long as the id is not in use by another \
            nnunet dataset",
    )
    parser.add_argument(
        "--fold", type=int, required=True, help="Fold of the 5-fold cross-validation. Should be an int between 0 and 4"
    )
    parser.add_argument(
        "--configs",
        type=str,
        nargs="+",
        required=True,
        help="Configuration that should be trained. Eg. '2d', '3d_fullres'."
        "Multiple configs can be passed here and all will be trained "
        "sequentially in the order provided given that a pretrained checkpoint"
        " is given for each config. Note that 3d_cascade_fullres requires that"
        " 3d_lowres already be trained first",
    )
    parser.add_argument(
        "--pretrain_checkpoints",
        type=str,
        required=True,
        nargs="+",
        help="Path to the model checkpoints of the pretrained models."
        " One checkpoint must be provided for each config and the order must be"
        " consistent with the --configs flag",
    )
    parser.add_argument(
        "--pretrain_plans",
        type=str,
        required=False,
        help="[OPTIONAL] The path to the nnUNet plans used for pretraining. \
            Defaults to looking for nnUNetPlans.json in the pretraining \
            datasets nnUNet_preprocessed folder",
    )
    parser.add_argument(
        "-num_processes",
        type=int,
        nargs="+",
        required=False,
        help="[OPTIONAL] Use this to define how many processes are to be used. If this is just one number then "
        "this number of processes is used for all configurations specified with --configs. If it's a "
        "list of numbers this list must have as many elements as there are configurations. We "
        "then iterate over zip(configs, num_processes) to determine then umber of processes "
        "used for each configuration. More processes is always faster (up to the number of "
        "threads your PC can support, so 8 for a 4 core CPU with hyperthreading. If you don't "
        "know what that is then dont touch it, or at least don't increase it!). DANGER: More "
        "often than not the number of processes that can be used is limited by the amount of "
        "RAM available. Image resampling takes up a lot of RAM. MONITOR RAM USAGE AND "
        "DECREASE -np IF YOUR RAM FILLS UP TOO MUCH!. Default: 8 processes for 2d, 4 "
        "for 3d_fullres, 8 for 3d_lowres and 4 for everything else",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        required=False,
        help="[OPTIONAL] Use this to set the device the training should run with. Available options are 'cuda' "
        "(GPU), 'cpu' (CPU) and 'mps' (Apple M1/M2). Do NOT use this to set which GPU ID! "
        "Use CUDA_VISIBLE_DEVICES=X nnUNetv2_train [...] instead!",
    )
    parser.add_argument(
        "--skip_preprocessing",
        action="store_true",
        required=False,
        help="[OPTIONAL] Use this to skip the preprocessing step if you have"
        "already run this script with preprocessing for the given pretrained model"
        "and finetuning dataset",
    )
    parser.add_argument(
        "--trainer",
        type=str,
        default="nnUNetTrainer",
        required=False,
        help="[OPTIONAL] The name of the nnunet trainer class to use. The default"
        "is nnUNetTrainer but nnUNet provides alternatives such as nnUNetTrainer_1epoch",
    )

    args = parser.parse_args()

    # If pretrain plans path not given, set default pretrain plans path
    if args.pretrain_plans is None:
        pt_dataset_name = convert_id_to_dataset_name(args.pretrain_id)
        args.pretrain_plans = os.path.join(nnUNet_preprocessed, pt_dataset_name, "nnUNetPlans.json")

    # Ensure all configs provided are in the original pretraining plans
    check_configs(plans_path=args.pretrain_plans, configs=args.configs)

    # Ensure a checkpoint was provided for each config
    assert len(args.configs) == len(args.pretrain_checkpoints), (
        f"The number of configs ({len(args.configs)}) and number of pretrained checkpoints "
        f"({len(args.pretrain_checkpoints)}) do not match"
    )

    # Transfer Metadata needed for processing the finetuning set
    ft_plans_identifier = transfer_metadata(
        finetune_id=args.finetune_id, pretrain_id=args.pretrain_id, pretrain_plans=args.pretrain_plans
    )

    # Set default number of processes for preprocessing
    if args.num_processes is None:
        default_num_processes = {"2d": 8, "3d_fullres": 4, "3d_lowres": 8}
        num_processes = [default_num_processes.get(c, 4) for c in args.configs]
    else:
        num_processes = args.num_processes

    # preprocessing
    if not args.skip_preprocessing:
        preprocess_dataset(
            dataset_id=args.finetune_id,
            plans_identifier=ft_plans_identifier,
            num_processes=num_processes,
            configurations=args.configs,
        )

    device = setup_device(args.device)
    # We enforce the models to train in the below order. Namely because 3d_lowres
    # must be trained before 3d_cascade_fullres. We do not currently check if
    # the given configs were present in the plans
    possible_configs = ["2d", "3d_fullres", "3d_lowres", "3d_cascade_fullres"]
    for config in possible_configs:
        if config in args.configs:
            i = args.configs.index(config)
            pt_ckpt = args.pretrain_checkpoints[i]
            kwargs = {
                "dataset_name_or_id": str(args.finetune_id),
                "configuration": config,
                "fold": args.fold,
                "trainer_class_name": args.trainer,
                "plans_identifier": ft_plans_identifier,
                "pretrained_weights": pt_ckpt,
                "device": device,
            }
            p = Process(target=run_training, kwargs=kwargs)
            print(f"Starting training of {config}")
            p.start()
            p.join()
            print("Training Done")


if __name__ == "__main__":
    main()
