import argparse
import os
import shutil
from multiprocessing import Process

import torch
from batchgenerators.utilities.file_and_folder_operations import load_json, save_json
from nnunetv2.experiment_planning.plan_and_preprocess_api import extract_fingerprints, preprocess_dataset
from nnunetv2.experiment_planning.plans_for_pretraining.move_plans_between_datasets import move_plans_between_datasets
from nnunetv2.paths import nnUNet_preprocessed, nnUNet_raw
from nnunetv2.run.run_training import run_training
from nnunetv2.utilities.dataset_name_id_conversion import convert_id_to_dataset_name


def setup_device(device: str) -> torch.device:
    """Checks the device being used, sets the number of threads and returns a pytorch device

    Args:
        device: Either 'cpu', 'cuda' or 'mps'

    Returns
        torch_device: The corresponding pytorch.device object
    """
    assert device in [
        "cpu",
        "cuda",
        "mps",
    ], f"-device must be either cpu, mps or cuda. Other devices are not tested/supported. Got: {device}."
    if device == "cpu":
        # let's allow torch to use hella threads
        import multiprocessing

        torch.set_num_threads(multiprocessing.cpu_count())
        return torch.device("cpu")
    elif device == "cuda":
        # multithreading in torch doesn't help nnU-Net if run on GPU
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        return torch.device("cuda")
    else:
        return torch.device("mps")


def transfer_metadata(ft_id: int, pt_id: int, pt_plans_identifier: str) -> str:
    """Transfers and or modifies metadata necessary for transfer learning

    Ensures the dataset json, dataset fingerprint and nnunet plans are all
    ready for transfer learning. Some necessary changes are made to the plans
    from pretraining to allow finetuning to occur without issue. Note that
    this function saves the plans for finetuning as 'pt_plans_identifier-ftXptY'
    where X is ft_if and Y is pt_id. If a plan file is already found that
    matches this pattern, the function does not overwrite it.

    Args:
        ft_id: The nnunet id of the finetuning dataset
        pt_id: The nnunet id of the pretraining dataset
        pt_plans_identifier: The identifier/name of the plans from pretraining
            to use as a basis for the finetuning plans
    Returns:
        tl_plans_identifier: The identifier for the plans to use during finetuning
    """
    # Extract Fingerprint if needed
    ft_dataset_name = convert_id_to_dataset_name(ft_id)
    fingerprint_path = os.path.join(nnUNet_preprocessed, ft_dataset_name, "dataset_fingerprint.json")
    if not os.path.isfile(fingerprint_path):
        extract_fingerprints(dataset_ids=[ft_id])
    else:
        print("INFO: The finetuning dataset fingerprint has already been extracted. Skipping fingerprint extraction")

    # Transfer pretrained model plans if needed
    tl_plans_identifier = pt_plans_identifier + "-pt%03.0d" % pt_id
    tl_plans_path = os.path.join(nnUNet_preprocessed, ft_dataset_name, tl_plans_identifier + ".json")
    if not os.path.isfile(tl_plans_path):
        # This automatically changes the dataset name and data identifier fields
        # of the plans file to match the finetuning/target dataset
        move_plans_between_datasets(
            source_dataset_name_or_id=pt_id,
            target_dataset_name_or_id=ft_id,
            source_plans_identifier=pt_plans_identifier,
            target_plans_identifier=tl_plans_identifier,
        )

        # The batch size needs to be adjusted for the new finetuning dataset as
        # its upper bound is determined by the number of samples in the dataset
        num_tr = load_json(os.path.join(nnUNet_raw, ft_dataset_name, "dataset.json"))["numTraining"]
        bs_5percent = round(num_tr * 0.05)  # The upper bound for batch size
        tl_plans = load_json(tl_plans_path)
        for c in tl_plans["configurations"]:  # Must check batch size for each config
            pt_bs = tl_plans["configurations"][c]["batch_size"]
            # Ensure batch size is below upper bound and above lower bound of 2
            bs = max(min(pt_bs, bs_5percent), 2)
            tl_plans["configurations"][c]["batch_size"] = bs
        save_json(tl_plans, tl_plans_path, sort_keys=False)
    else:
        print("INFO: The {tl_plans_identifier} plans file already exists. Using that instead")

    # Copy dataset json if needed
    if not os.path.isfile(os.path.join(nnUNet_preprocessed, ft_dataset_name, "dataset.json")):
        shutil.copy(
            os.path.join(nnUNet_raw, ft_dataset_name, "dataset.json"),
            os.path.join(nnUNet_preprocessed, ft_dataset_name, "dataset.json"),
        )

    return tl_plans_identifier


def check_configs(id: int, plans_identifier: str, configs: list) -> None:
    """Raises an error if configs are not found in plans json

    Args:
        id: The nnunet id of the dataset for which the plans were created
        plans_identifier: The name of the plans file to check, not including
            any extenstions
        configs: A list of nnunet configs to check for
    """
    dataset_name = convert_id_to_dataset_name(id)
    plans_path = os.path.join(nnUNet_preprocessed, dataset_name, plans_identifier + ".json")
    plans = load_json(plans_path)
    for c in configs:
        assert (
            c in plans["configurations"]
        ), f"Did not find {c} config in {plans_identifier}.json. Note that \
            3d_lowres and 3d_cascade_fullres are not created for datasets with \
            small image sizes"


def main() -> None:
    """Parses arguments and opens subprocesses to train specified models

    Notes: Subqequent trainings after the first model training must run in
        subprocesses otherwise pytorch will refuse to use gpu for some reason.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ft_id", type=int, required=True, help="The nnUNet dataset ID integer for the fine-tuning dataset"
    )
    parser.add_argument(
        "--pt_id",
        type=int,
        required=True,
        help="[OPTIONAL] The dataset ID integer of the pretraining dataset"
        ". Can be omitted if pt_fingerprint, pt_plans and pt_identifier"
        "flags are all included",
    )
    parser.add_argument("--pt_plans_identifier", type=str, required=False, default="nnUNetPlans")
    parser.add_argument(
        "--pt_checkpoints",
        type=str,
        required=True,
        nargs="+",
        help="Path to the model checkpoints of the pretrained models."
        " One checkpoint must be provided for each config and the order must be"
        " consistent with the --configs flag",
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
        "--fold", type=int, required=True, help="Fold of the 5-fold cross-validation. Should be an int between 0 and 4"
    )
    parser.add_argument(
        "-np",
        type=int,
        nargs="+",
        default=[8, 4, 8],
        required=False,
        help="[OPTIONAL] Use this to define how many processes are to be used. If this is just one number then "
        "this number of processes is used for all configurations specified with -c. If it's a "
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
        "-device",
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

    args = parser.parse_args()

    check_configs(id=args.pt_id, plans_identifier=args.pt_plans_identifier, configs=args.configs)

    assert len(args.configs) == len(
        args.pt_checkpoints
    ), f"The number of configs ({len(args.configs)}) and number of pretrained \
        checkpoints ({len(args.pt_checkpoints)}) do not match"

    # Transfer Metadata needed for processing the finetuning set
    tl_plans_identifier = transfer_metadata(
        ft_id=args.ft_id, pt_id=args.pt_id, pt_plans_identifier=args.pt_plans_identifier
    )

    # Set default number of processes for preprocessing
    if args.np is None:
        default_np = {"2d": 8, "3d_fullres": 4, "3d_lowres": 8}
        np = [default_np[c] if c in default_np.keys() else 4 for c in args.configs]
    else:
        np = args.np

    # preprocessing
    if not args.skip_preprocessing:
        preprocess_dataset(dataset_id=args.ft_id, plans_identifier=tl_plans_identifier, num_processes=np)

    device = setup_device(args.device)
    # We enforce the models to train in the below order. Namely because 3d_lowres
    # must be trained before 3d_cascade_fullres. We do not currently check if
    # the given configs were present in the plans
    possible_configs = ["2d", "3d_fullres", "3d_lowres", "3d_cascade_fullres"]
    for config in possible_configs:
        if config in args.configs:
            i = args.configs.index(config)
            pt_ckpt = args.pt_checkpoints[i]
            kwargs = {
                "dataset_name_or_id": str(args.ft_id),
                "configuration": config,
                "fold": args.fold,
                "trainer_class_name": "nnUNetTrainer_1epoch",
                "plans_identifier": tl_plans_identifier,
                "pretrained_weights": pt_ckpt,
                "device": device,
            }
            p = Process(target=run_training, kwargs=kwargs)
            print("Starting training of {config}")
            p.start()
            p.join()
            print("Training Done")


if __name__ == "__main__":
    main()
