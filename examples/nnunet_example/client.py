import argparse
import os
from logging import DEBUG, INFO
from os.path import exists, join
from pathlib import Path

import torch
from flwr.client import start_client
from flwr.common.logger import log, update_console_handler
from nnunetv2.dataset_conversion.convert_MSD_dataset import convert_msd_dataset

from fl4health.checkpointing.checkpointer import PerRoundStateCheckpointer
from fl4health.checkpointing.client_module import ClientCheckpointAndStateModule
from fl4health.clients.nnunet_client import NnunetClient
from fl4health.utils.load_data import load_msd_dataset
from fl4health.utils.metrics import EMAMetric, HardDice
from fl4health.utils.msd_dataset_sources import get_msd_dataset_enum
from fl4health.utils.nnunet_utils import set_nnunet_env


def main(
    dataset_path: Path,
    msd_dataset_name: str,
    server_address: str,
    fold: int | str,
    always_preprocess: bool = False,
    verbose: bool = True,
    compile: bool = True,
    intermediate_client_state_dir: str | None = None,
    client_name: str | None = None,
) -> None:
    # Log device and server address
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(INFO, f"Using device: {device}")
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

    # Create a metrics hard dice metric
    # NnunetClient automatically ensures that preds and targets are one-hot-encoded
    # HardDice will binarize the preds along the channel dimension for us using an argmax.
    hard_dice = HardDice("DICE", along_axes=(1,), ignore_background_axis=1, ignore_null=True, binarize=int(1))
    ema_dice = EMAMetric(metric=hard_dice, smoothing_factor=0.1)

    # For state checkpointing. Allows training to be resumed from state checkpoint
    if intermediate_client_state_dir is not None:
        checkpoint_and_state_module = ClientCheckpointAndStateModule(
            state_checkpointer=PerRoundStateCheckpointer(Path(intermediate_client_state_dir))
        )
    else:
        checkpoint_and_state_module = None

    # Create client
    client = NnunetClient(
        # Args specific to nnUNetClient
        dataset_id=dataset_id,
        fold=fold,
        always_preprocess=always_preprocess,
        verbose=verbose,
        compile=compile,
        # BaseClient Args
        device=device,
        metrics=[hard_dice, ema_dice],
        progress_bar=verbose,
        checkpoint_and_state_module=checkpoint_and_state_module,
        client_name=client_name,
    )

    start_client(server_address=server_address, client=client.to_client())

    # Shutdown the client
    client.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="nnunet_example/client.py",
        description="An example of nnUNetClient on any of the Medical \
            Segmentation Decathlon (MSD) datasets. Automatically generates a \
            nnunet segmentation model and trains it in a federated setting",
    )

    # I have to use underscores instead of dashes because thats how they
    # defined it in run_smoke_tests
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the folder in which data should be stored. This script \
            will automatically create nnunet_raw, and nnunet_preprocessed \
            subfolders if they don't already exist. This script will also \
            attempt to download and prepare the MSD Dataset into the \
            nnunet_raw folder if it does not already exist.",
    )
    parser.add_argument(
        "--fold",
        type=str,
        required=False,
        default="0",
        help="[OPTIONAL] Which fold of the local client dataset to use for \
            validation. nnunet defaults to 5 folds (0 to 4). Can also be set \
            to 'all' to use all the data for both training and validation. \
            Defaults to fold 0",
    )
    parser.add_argument(
        "--msd_dataset_name",
        type=str,
        required=False,
        default="Task04_Hippocampus",  # The smallest dataset
        help="[OPTIONAL] Name of the MSD dataset to use. The options are \
            defined by the values of the MsdDataset enum as returned by the \
            get_msd_dataset_enum function",
    )
    parser.add_argument(
        "--always-preprocess",
        action="store_true",
        required=False,
        help="[OPTIONAL] Use this to force preprocessing the nnunet data \
            even if the preprocessed data is found to already exist",
    )
    parser.add_argument(
        "--server_address",
        type=str,
        required=False,
        default="0.0.0.0:8080",
        help="[OPTIONAL] The server address for the clients to communicate \
            to the server through. Defaults to 0.0.0.0:8080",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        required=False,
        help="[OPTIONAL] Use this flag to see extra INFO logs and a progress bar",
    )
    parser.add_argument(
        "--debug",
        help="[OPTIONAL] Include flag to print DEBUG logs",
        action="store_const",
        dest="logLevel",
        const=DEBUG,
        default=INFO,
    )
    parser.add_argument(
        "--skip-compile",
        action="store_true",
        required=False,
        help="[OPTIONAL] Include flag to train without jit compiling the pytorch model first",
    )

    parser.add_argument(
        "--intermediate-client-state-dir",
        type=str,
        required=False,
        default=None,
        help="[OPTIONAL] Directory to store client state during training. Defaults to None",
    )
    parser.add_argument(
        "--client-name",
        type=str,
        required=False,
        default=None,
        help="[OPTIONAL] Name of the client used to name client state checkpoint. \
        Defaults to None, in which case a random name is generated for the client",
    )

    args = parser.parse_args()

    # Set the log level
    update_console_handler(level=args.logLevel)

    # Create nnunet directory structure and set environment variables
    nnUNet_raw = join(args.dataset_path, "nnunet_raw")
    nnUNet_preprocessed = join(args.dataset_path, "nnunet_preprocessed")
    os.makedirs(nnUNet_raw, exist_ok=True)
    os.makedirs(nnUNet_preprocessed, exist_ok=True)
    set_nnunet_env(
        nnUNet_raw=nnUNet_raw,
        nnUNet_preprocessed=nnUNet_preprocessed,
        nnUNet_results=join(args.dataset_path, "nnUNet_results"),
    )

    # Check fold argument and start main method
    fold: int | str = "all" if args.fold == "all" else int(args.fold)
    main(
        dataset_path=Path(args.dataset_path),
        msd_dataset_name=args.msd_dataset_name,
        server_address=args.server_address,
        fold=fold,
        always_preprocess=args.always_preprocess,
        verbose=args.verbose,
        compile=not args.skip_compile,
        intermediate_client_state_dir=args.intermediate_client_state_dir,
        client_name=args.client_name,
    )
