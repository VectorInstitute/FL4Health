import argparse
import os
import warnings
from logging import INFO
from os.path import exists, join
from pathlib import Path
from typing import Union

with warnings.catch_warnings():
    # Need to import lightning utilities now in order to avoid deprecation
    # warnings. Ignore flake8 warning saying that it is unused
    # lightning utilities is imported by some of the dependencies
    # so by importing it now and filtering the warnings
    # https://github.com/Lightning-AI/utilities/issues/119
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import lightning_utilities  # noqa: F401

import torch
from flwr.client import start_client
from flwr.common.logger import log
from torchmetrics.segmentation import GeneralizedDiceScore

from fl4health.utils.load_data import load_msd_dataset
from fl4health.utils.metrics import TorchMetric, TransformsMetric
from fl4health.utils.msd_dataset_sources import get_msd_dataset_enum, msd_num_labels
from research.picai.fl_nnunet.transforms import get_annotations_from_probs, get_probabilities_from_logits


def main(
    dataset_path: Path,
    msd_dataset_name: str,
    server_address: str,
    fold: Union[int, str],
    always_preprocess: bool = False,
) -> None:

    # Log device and server address
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(INFO, f"Using device: {DEVICE}")
    log(INFO, f"Using server address: {server_address}")

    # Load the dataset if necessary
    msd_dataset_enum = get_msd_dataset_enum(msd_dataset_name)
    nnUNet_raw = join(dataset_path, "nnunet_raw")
    if not exists(join(nnUNet_raw, msd_dataset_enum.value)):
        log(INFO, f"Downloading and extracting {msd_dataset_enum.value} dataset")
        load_msd_dataset(nnUNet_raw, msd_dataset_name)

    # The dataset ID will be the same as the MSD Task number
    dataset_id = int(msd_dataset_enum.value[4:6])
    nnunet_dataset_name = f"Dataset{dataset_id:03d}_{msd_dataset_enum.value.split('_')[1]}"

    # Convert the msd dataset if necessary
    if not exists(join(nnUNet_raw, nnunet_dataset_name)):
        log(INFO, f"Converting {msd_dataset_enum.value} into nnunet dataset")
        convert_msd_dataset(source_folder=join(nnUNet_raw, msd_dataset_enum.value))

    # Create a metric
    dice = TransformsMetric(
        metric=TorchMetric(
            name="Pseudo DICE",
            metric=GeneralizedDiceScore(
                num_classes=msd_num_labels[msd_dataset_enum], weight_type="square", include_background=False
            ).to(DEVICE),
        ),
        transforms=[get_probabilities_from_logits, get_annotations_from_probs],
    )

    # Create client
    client = nnUNetClient(
        # Args specific to nnUNetClient
        dataset_id=dataset_id,
        fold=fold,
        always_preprocess=always_preprocess,
        # BaseClient Args
        device=DEVICE,
        metrics=[dice],
        data_path=dataset_path,  # Argument not actually used by nnUNetClient
    )

    start_client(server_address=server_address, client=client.to_client())

    # Shutdown the client
    client.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="nnunet_example/client.py",
        description="""An exampled of nnUNetClient on any of the Medical
            Segmentation Decathelon (MSD) datasets. Automatically generates a
            nnunet segmentation model and trains it in a federated setting""",
    )

    # I have to use underscores instead of dashes because thats how they
    # defined it in run smoke tests
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="""Path to the folder in which data should be stored. This script
            will automatically create nnunet_raw, and nnunet_preprocessed
            subfolders if they don't already exist. This script will also
            attempt to download and prepare the MSD Dataset into the
            nnunet_raw folder if it does not already exist.""",
    )
    parser.add_argument(
        "--fold",
        type=str,
        required=False,
        default="0",
        help="""[OPTIONAL] Which fold of the local client dataset to use for
            validation. nnunet defaults to 5 folds (0 to 4). Can also be set
            to 'all' to use all the data for both training and validation.
            Defaults to fold 0""",
    )
    parser.add_argument(
        "--msd_dataset_name",
        type=str,
        required=False,
        default="Task04_Hippocampus",  # The smallest dataset
        help="""[OPTIONAL] Name of the MSD dataset to use. The options are
            defined by the values of the MsdDataset enum as returned by the
            get_msd_dataset_enum function""",
    )
    parser.add_argument(
        "--always-preprocess",
        action="store_true",
        required=False,
        help="""[OPTIONAL] Use this to force preprocessing the nnunet data
            even if the preprocessed data is found to already exist""",
    )
    parser.add_argument(
        "--server_address",
        type=str,
        required=False,
        default="0.0.0.0:8080",
        help="""[OPTIONAL] The server address for the clients to communicate
            to the server through. Defaults to 0.0.0.0:8080""",
    )
    args = parser.parse_args()

    # Create nnunet directory structure and set environment variables
    nnUNet_raw = join(args.dataset_path, "nnunet_raw")
    nnUNet_preprocessed = join(args.dataset_path, "nnunet_preprocessed")
    if not exists(nnUNet_raw):
        os.makedirs(nnUNet_raw)
    if not exists(nnUNet_preprocessed):
        os.makedirs(nnUNet_preprocessed)
    os.environ["nnUNet_raw"] = nnUNet_raw
    os.environ["nnUNet_preprocessed"] = nnUNet_preprocessed
    os.environ["nnUNet_results"] = join(args.dataset_path, "nnunet_results")
    log(INFO, "Setting nnunet environment variables")
    log(INFO, f"\tnnUNet_raw: {nnUNet_raw}")
    log(INFO, f"\tnnUNet_preprocessed: {nnUNet_preprocessed}")
    log(INFO, f"\tnnUNet_results: {join(args.dataset_path, 'nnunet_results')}")

    # Everything that uses nnunetv2 module can only be imported after
    # environment variables are changed
    from nnunetv2.dataset_conversion.convert_MSD_dataset import convert_msd_dataset

    from research.picai.fl_nnunet.nnunet_client import nnUNetClient

    # Check fold argument and start main method
    fold: Union[int, str] = "all" if args.fold == "all" else int(args.fold)
    main(
        dataset_path=Path(args.dataset_path),
        msd_dataset_name=args.msd_dataset_name,
        server_address=args.server_address,
        fold=fold,
        always_preprocess=args.always_preprocess,
    )
