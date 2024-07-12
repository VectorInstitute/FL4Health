import warnings

with warnings.catch_warnings():
    # Need to import lightning utilities now in order to avoid warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import lightning_utilities

import argparse
from logging import INFO
from os.path import join
from typing import Literal, Optional

warnings.filterwarnings("ignore", category=DeprecationWarning)
import flwr as fl
import torch
from flwr.common.logger import log
from nnunetv2.paths import nnUNet_preprocessed
from nnunetv2.utilities.dataset_name_id_conversion import convert_id_to_dataset_name
from torchmetrics.classification import Dice
from torchmetrics.segmentation import GeneralizedDiceScore

from fl4health.utils.metrics import TorchMetric, TransformsMetric
from research.picai.fl_nnunet.nnunet_client import nnUNetClient
from research.picai.fl_nnunet.transforms import collapse_ohe_target, get_ann_from_probs, get_prob_from_logits

nnUNetConfig = Literal["2d", "3d_fullres", "3d_lowres", "3d_cascade_fullres"]


def main(
    dataset_id: int,
    data_identifier: Optional[str],
    plans_identifier: Optional[str],
    always_preprocess: bool,
    server_address: str,
) -> None:

    # Log device and server address
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(INFO, f"Using device: {DEVICE}")
    log(INFO, f"Using server Address: {server_address}")

    # Define metrics
    dice1 = TransformsMetric(
        metric=TorchMetric(
            name="dice1",
            metric=GeneralizedDiceScore(num_classes=2, weight_type="square", include_background=False).to(DEVICE),
        ),
        transforms=[get_prob_from_logits, get_ann_from_probs],
    )
    # The Dice class requires preds to be ohe, but targets to not be ohe
    dice2 = TransformsMetric(
        metric=TorchMetric(name="dice2", metric=Dice(num_classes=2, ignore_index=0).to(DEVICE)),
        transforms=[get_prob_from_logits, collapse_ohe_target],
    )

    metrics = [dice1, dice2]  # Oddly each of these dice metrics is drastically different.

    # Create and start client
    dataset_name = convert_id_to_dataset_name(dataset_id)
    client = nnUNetClient(
        # Args specific to nnUNetClient
        dataset_id=dataset_id,
        data_identifier=data_identifier,
        plans_identifier=plans_identifier,
        always_preprocess=always_preprocess,
        # BaseClient Args
        device=DEVICE,
        metrics=metrics,  # Had to turn metrics off for now because for some reason nnunet model predicts 2 channels
        data_path=join(
            nnUNet_preprocessed, dataset_name
        ),  # data_path is not actually used but is required by BaseClient
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
        help="[OPTIONAL ]Include this flag to have the client save the plans \
            file that it uses for training locally. The user can also pass an \
            argument to this flag to specify what the plans file should be \
            names. Defaults to 'FL_Dataset123_plansname' where plansname \
            is the name of the plans file sent to the client by the server",
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
    main(
        dataset_id=args.dataset_id,
        data_identifier=args.data_identifier,
        plans_identifier=args.plans_identifier,
        always_preprocess=args.always_preprocess,
        server_address=args.server_address,
    )
