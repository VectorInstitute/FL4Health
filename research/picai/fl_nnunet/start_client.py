import argparse
import warnings
from logging import INFO
from pathlib import Path
from typing import Optional, Union

with warnings.catch_warnings():
    # Need to import lightning utilities now in order to avoid deprecation
    # warnings. Ignore flake8 warning saying that it is unused
    # lightning utilities is imported by some of the dependencies
    # so by importing it now and filtering the warnings
    # https://github.com/Lightning-AI/utilities/issues/119
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import lightning_utilities  # noqa: F401

import flwr as fl
import torch
from flwr.common.logger import log
from torchmetrics.classification import Dice
from torchmetrics.segmentation import GeneralizedDiceScore

from fl4health.utils.metrics import TorchMetric, TransformsMetric
from research.picai.fl_nnunet.nnunet_client import nnUNetClient
from research.picai.fl_nnunet.transforms import (
    collapse_one_hot_target,
    get_annotations_from_probs,
    get_probabilities_from_logits,
)


def main(
    dataset_id: int,
    data_identifier: Optional[str],
    plans_identifier: Optional[str],
    always_preprocess: bool,
    server_address: str,
    fold: Union[str, int],
) -> None:

    # Log device and server address
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(INFO, f"Using device: {DEVICE}")
    log(INFO, f"Using server address: {server_address}")

    # Define metrics
    dice1 = TransformsMetric(
        metric=TorchMetric(
            name="dice1",
            metric=GeneralizedDiceScore(num_classes=2, weight_type="square", include_background=False).to(DEVICE),
        ),
        transforms=[get_probabilities_from_logits, get_annotations_from_probs],
    )
    # The Dice class requires preds to be ohe, but targets to not be ohe
    dice2 = TransformsMetric(
        metric=TorchMetric(name="dice2", metric=Dice(num_classes=2, ignore_index=0).to(DEVICE)),
        transforms=[get_probabilities_from_logits, collapse_one_hot_target],
    )

    metrics = [dice1, dice2]  # Oddly each of these dice metrics is drastically different.

    # Create and start client
    client = nnUNetClient(
        # Args specific to nnUNetClient
        dataset_id=dataset_id,
        fold=fold,
        data_identifier=data_identifier,
        plans_identifier=plans_identifier,
        always_preprocess=always_preprocess,
        # BaseClient Args
        device=DEVICE,
        metrics=metrics,
        progress_bar=False,
        data_path=Path("dummy/path"),  # Argument not used by nnUNetClient
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
            nnunet config (eg. 2d, 3d_fullres, etc.). If preprocessed data \
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

    args = parser.parse_args()

    # Convert fold to an integer if it is not 'all'
    fold: Union[int, str] = "all" if args.fold == "all" else int(args.fold)

    main(
        dataset_id=args.dataset_id,
        data_identifier=args.data_identifier,
        plans_identifier=args.plans_identifier,
        always_preprocess=args.always_preprocess,
        server_address=args.server_address,
        fold=fold,
    )
