import argparse
import os
import warnings
from logging import DEBUG, INFO
from os.path import exists, join
from pathlib import Path
from typing import Any

from fl4health.checkpointing.client_module import ClientCheckpointAndStateModule
from fl4health.checkpointing.state_checkpointer import ClientStateCheckpointer


with warnings.catch_warnings():
    # Silence deprecation warnings from sentry sdk due to flwr and wandb
    # https://github.com/adap/flower/issues/4086
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import wandb  # noqa: F401

import torch
from flwr.client import start_client
from flwr.common.logger import log, update_console_handler
from nnunetv2.dataset_conversion.convert_MSD_dataset import convert_msd_dataset

from fl4health.clients.flexible.nnunet import FlexibleNnunetClient
from fl4health.metrics.base_metrics import Metric
from fl4health.metrics.compound_metrics import EmaMetric
from fl4health.metrics.efficient_metrics import BinaryDice, MultiClassDice
from fl4health.mixins.personalized import PersonalizedMode, make_it_personal
from fl4health.utils.load_data import load_msd_dataset
from fl4health.utils.msd_dataset_sources import get_msd_dataset_enum, msd_num_labels
from fl4health.utils.nnunet_utils import set_nnunet_env
from fl4health.utils.random import set_all_random_seeds


N_CLASSES_2D = 2

personalized_client_classes = {
    "ditto": make_it_personal(FlexibleNnunetClient, PersonalizedMode.DITTO),
    "mr_mtl": make_it_personal(FlexibleNnunetClient, PersonalizedMode.MR_MTL),
}


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
    personalized_strategy: str = "ditto",
) -> None:
    # Log device and server address
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(INFO, f"Using device: {device}")
    log(INFO, f"Using server address: {server_address}")

    # Load the dataset if necessary
    msd_dataset_enum = get_msd_dataset_enum(msd_dataset_name)
    nnUNet_raw = join(dataset_path, "nnunet_raw")  # noqa: N806
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

    # Create dice metric
    dice: Metric
    if msd_num_labels[msd_dataset_enum] > N_CLASSES_2D:
        dice = MultiClassDice(
            batch_dim=None,  # Aggregate across all samples in batch/round
            label_dim=1,  # Separate dice for each output class
            name="Dice",
            threshold=1,  # Use an argmax to binarize output logits (unactivated)
            ignore_background=1,  # Ignore background class
        )
    else:  # Background class is automatically ignored for BinaryDice
        dice = BinaryDice(batch_dim=None, label_dim=1, name="Dice", threshold=1)

    # Create EMA Dice metric
    ema_dice = EmaMetric(dice)

    # State checkpointer (being overhauled soon)
    if intermediate_client_state_dir is not None:
        checkpoint_and_state_module = ClientCheckpointAndStateModule(
            state_checkpointer=ClientStateCheckpointer(Path(intermediate_client_state_dir))
        )
    else:
        checkpoint_and_state_module = None

    # Create client
    client_kwargs: dict[str, Any] = {
        # Args specific to nnUNetClient
        "dataset_id": dataset_id,
        "fold": fold,
        "always_preprocess": always_preprocess,
        "verbose": verbose,
        "compile": compile,
        # BaseClient Args
        "device": device,
        "metrics": [dice, ema_dice],
        "progress_bar": verbose,
        "checkpoint_and_state_module": checkpoint_and_state_module,
        "client_name": client_name,
    }
    if personalized_strategy in personalized_client_classes:
        log(
            INFO,
            f"Setting up client for personalized strategy: {personalized_strategy}",
        )
        client = personalized_client_classes[personalized_strategy](**client_kwargs)
    else:
        log(INFO, "Setting up client without personalization")
        client = FlexibleNnunetClient(**client_kwargs)
    log(INFO, f"Using client: {type(client).__name__}")

    start_client(server_address=server_address, client=client.to_client())

    # Shutdown the client
    client.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="nnunet_example/client.py",
        description="An exampled of nnUNetClient on any of the Medical \
            Segmentation Decathlon (MSD) datasets. Automatically generates a \
            nnunet segmentation model and trains it in a federated setting",
    )

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
    parser.add_argument(
        "--personalized_strategy",
        type=str,
        required=False,
        default="ditto",
        help="[OPTIONAL] Personalized strategy to use. For now, can only be 'ditto'. \
        Defaults to 'ditto'.",
    )
    parser.add_argument(
        "--seed",
        action="store",
        type=int,
        help="Seed for the random number generators across python, torch, and numpy",
        required=False,
    )

    args = parser.parse_args()

    # Set the random seed for reproducibility
    # NOTE: This implementation does not cover all sources of randomness in nnUNet, so complete
    # determinism cannot be achieved. The nnUNet maintainers have confirmed that full determinism
    # is not possible (see linked issue below). However, our current approach provides a reasonable
    # level of deterministic behavior for most practical purposes.
    # Reference: https://github.com/VectorInstitute/FL4Health/pull/411#:~:text=MIC%2DDKFZ/nnUNet%231906
    set_all_random_seeds(
        # NOTE: Setting seed comes at the cost of runtime performance. Benchmarking especially should be enabled
        # for long-running experiments.
        args.seed,
        disable_torch_benchmarking=True,
        use_deterministic_torch_algos=True,
    )
    # Set the log level
    update_console_handler(level=args.logLevel)

    # Create nnunet directory structure and set environment variables
    nn_unet_raw = join(args.dataset_path, "nnunet_raw")
    nn_unet_preprocessed = join(args.dataset_path, "nnunet_preprocessed")
    os.makedirs(nn_unet_raw, exist_ok=True)
    os.makedirs(nn_unet_preprocessed, exist_ok=True)
    set_nnunet_env(
        nnUNet_raw=nn_unet_raw,
        nnUNet_preprocessed=nn_unet_preprocessed,
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
        personalized_strategy=args.personalized_strategy,
    )
