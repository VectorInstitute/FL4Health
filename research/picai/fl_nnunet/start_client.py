import argparse
import logging
import warnings
from functools import partial
from logging import INFO
from pathlib import Path

from fl4health.checkpointing.client_module import ClientCheckpointAndStateModule
from fl4health.checkpointing.state_checkpointer import ClientStateCheckpointer


with warnings.catch_warnings():
    # Silence deprecation warnings from sentry sdk due to flwr and wandb
    # https://github.com/adap/flower/issues/4086
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import wandb  # noqa: F401

import flwr as fl
import torch
from flwr.common.logger import log, update_console_handler
from torchmetrics.segmentation import DiceScore, GeneralizedDiceScore

from fl4health.clients.nnunet_client import NnunetClient
from fl4health.metrics import TorchMetric
from fl4health.metrics.compound_metrics import TransformsMetric
from fl4health.utils.nnunet_utils import collapse_one_hot_tensor, get_segs_from_probs


def main(
    dataset_id: int,
    data_identifier: str | None,
    plans_identifier: str | None,
    always_preprocess: bool,
    server_address: str,
    fold: str | int,
    verbose: bool,
    compile: bool,
    intermediate_client_state_dir: str | None = None,
    client_name: str | None = None,
) -> None:
    # Log device and server address
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(INFO, f"Using device: {device}")
    log(INFO, f"Using server address: {server_address}")

    # Define metrics
    dice1 = TransformsMetric(
        metric=TorchMetric(
            name="dice1",
            metric=GeneralizedDiceScore(num_classes=2, weight_type="square", include_background=False).to(device),
        ),
        pred_transforms=[torch.sigmoid, get_segs_from_probs],
    )
    # The Dice class requires preds to be ohe, but targets to not be ohe
    dice2 = TransformsMetric(
        metric=TorchMetric(name="dice2", metric=DiceScore(num_classes=2, include_background=False).to(device)),
        pred_transforms=[torch.sigmoid],
        target_transforms=[partial(collapse_one_hot_tensor, dim=1)],
    )

    metrics = [dice1, dice2]  # Oddly each of these dice metrics is drastically different.

    if intermediate_client_state_dir is not None:
        checkpoint_and_state_module = ClientCheckpointAndStateModule(
            state_checkpointer=ClientStateCheckpointer(Path(intermediate_client_state_dir))
        )
    else:
        checkpoint_and_state_module = None

    # Create and start client
    client = NnunetClient(
        # Args specific to nnUNetClient
        dataset_id=dataset_id,
        fold=fold,
        data_identifier=data_identifier,
        plans_identifier=plans_identifier,
        always_preprocess=always_preprocess,
        verbose=verbose,
        compile=compile,
        # BaseClient Args
        device=device,
        metrics=metrics,
        progress_bar=verbose,
        checkpoint_and_state_module=checkpoint_and_state_module,
        client_name=client_name,
    )

    fl.client.start_client(server_address=server_address, client=client.to_client())

    # Shutdown the client
    client.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-id", type=int, required=True, help="The nnunet dataset id for the local client training dataset"
    )
    parser.add_argument(
        "--fold",
        type=str,
        required=True,
        help="""Which fold of the local client dataset to use for validation.
            nnunet defaults to 5 folds (0 to 4). Can also be set to 'all' to
            use all the data for both training and validation.""",
    )
    parser.add_argument(
        "--data-identifier",
        type=str,
        required=False,
        help="[OPTIONAL] The nnunet data identifier prefix to use. The final \
            data identifier will be the 'prefix_config' where config is the \
            nnunet config (e.g. 2d, 3d_fullres, etc.). If preprocessed data \
            already exists can be used to specify which preprocessed data to \
            use. The default prefix is the plans name used during training (see \
            the --save-plans flag)",
    )
    parser.add_argument(
        "--plans-identifier",
        required=False,
        help="[OPTIONAL ] Specify what the plans file should be \
            named. Defaults to 'FL_Dataset000_plansname' where plansname \
            is the name of the plans file sent to the client by the server \
            and 000 is the nnunet dataset id of the local client",
    )
    parser.add_argument(
        "--always-preprocess",
        action="store_true",
        required=False,
        help="[OPTIONAL] Use this to force preprocessing the nnunet data even \
            if the preprocessed data is found to already exist",
    )
    parser.add_argument(
        "--server-address",
        type=str,
        required=False,
        default="0.0.0.0:8080",
        help="The server address for the clients to communicate to the server \
            through. Defaults to 0.0.0.0:8080",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        required=False,
        help="[OPTIONAL] Use this flag to log extra INFO logs and a progress bar",
    )
    parser.add_argument(
        "--debug",
        help="Include flag to print debug logs",
        action="store_const",
        dest="logLevel",
        const=logging.DEBUG,
        default=logging.INFO,
    )
    parser.add_argument(
        "--quiet",
        help="Include flag to silence INFO and DEBUG logs",
        action="store_const",
        dest="logLevel",
        const=logging.WARNING,
    )
    parser.add_argument(
        "--skip-compile",
        action="store_true",
        required=False,
        help="[OPTIONAL] Include flag to train without jit compiling the pytorch model first",
    )
    parser.add_argument(
        "--intermediate-client-state-dir",
        default="./",
        required=False,
        help="[OPTIONAL] Directory to checkpoint client state. Defaults to current directory.",
    )
    parser.add_argument("--client-name", required=False, help="[OPTIONAL] Name of client.")

    args = parser.parse_args()

    # Set log level
    update_console_handler(level=args.logLevel)

    # Convert fold to an integer if it is not 'all'
    fold: int | str = "all" if args.fold == "all" else int(args.fold)

    main(
        dataset_id=args.dataset_id,
        data_identifier=args.data_identifier,
        plans_identifier=args.plans_identifier,
        always_preprocess=args.always_preprocess,
        server_address=args.server_address,
        fold=fold,
        verbose=args.verbose,
        compile=not args.skip_compile,
        intermediate_client_state_dir=args.intermediate_client_state_dir,
        client_name=args.client_name,
    )
