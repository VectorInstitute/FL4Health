import pickle
from logging import DEBUG, INFO, WARNING
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from flwr.common.logger import log
from flwr.common.typing import Config, List, NDArrays, Scalar

from fl4health.checkpointing.checkpointer import TorchCheckpointer
from fl4health.clients.basic_client import BasicClient
from fl4health.security.secure_aggregation import ClientCryptoKit, Event
from fl4health.utils.losses import Losses, LossMeter, LossMeterType
from fl4health.utils.metrics import Metric, MetricMeter, MetricMeterManager, MetricMeterType


class SecureAggregationClient(BasicClient):
    def __init__(
        self,
        data_path: Path,
        metrics: Sequence[Metric],
        device: torch.device,
        loss_meter_type: LossMeterType = LossMeterType.AVERAGE,
        metric_meter_type: MetricMeterType = MetricMeterType.AVERAGE,
        checkpointer: Optional[TorchCheckpointer] = None,
    ) -> None:
        super().__init__(data_path, metrics, device, loss_meter_type, metric_meter_type, checkpointer)

        # handles SecAgg cryptography on the client-side
        self.crypto = ClientCryptoKit()

        # federated round
        self.fl_round = 0

    def fit(self, parameters: NDArrays, config: Config) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        local_epochs, local_steps, current_server_round = self.process_config(config)

        if not self.initialized:
            self.setup_client(config)

        self.set_parameters(parameters, config)
        if local_epochs is not None:
            loss_dict, metrics = self.train_by_epochs(local_epochs, current_server_round)
            local_steps = len(self.train_loader) * local_epochs  # total steps over training round
        elif local_steps is not None:
            loss_dict, metrics = self.train_by_steps(local_steps, current_server_round)
        else:
            raise ValueError("Must specify either local_epochs or local_steps in the Config.")

        # Update after train round (Used by Scaffold and DP-Scaffold Client to update control variates)
        self.update_after_train(local_steps, loss_dict)

        # FitRes should contain local parameters, number of examples on client, and a dictionary holding metrics
        # calculation results.

        # add mask and privacy noise
        self.modify_parmeters()

        return (
            self.get_parameters(config),
            self.num_train_samples,
            metrics,
        )

    def modify_parmeters(self):
        "Use this method to post process model parameters (i.e. masking & noising) after train and distributed evaluation."

        # counts trainable model parameters
        dim = sum(param.numel() for param in self.model.parameters() if param.requires_grad)

        # computes masking vector; this can only be run after masking seed agreement
        pair_mask_vect: List[int] = self.crypto.get_pair_mask_sum(
            vector_dim=dim
        )  # pass in online_clients kwarg for drop out case

        # modify parms  << TODO quantization for privacy mechanism + modular arithmetic>>
        i = 0
        params_dict = self.model.state_dict()
        for name, params in params_dict.items():
            j = i + params.numel()
            mask = torch.tensor(pair_mask_vect[i:j]).reshape(params.shape)
            params_dict[name] *= 0  # TEST ONLY: remove zeroing of params
            self.debugger(params_dict[name])
            params_dict[name] += mask
            self.debugger(params_dict[name])
            i = j

        # load modified parms
        self.model.load_state_dict(params_dict)

    def get_properties(self, config: Config) -> Dict[str, Scalar]:
        if not self.initialized:
            self.setup_client(config)
        # be sure to include a round in config
        # self.fl_round = config["fl_round"]

        response_dict = {}

        match config["event_name"]:
            case Event.ADVERTISE_KEYS.value:
                assert config["sender"] == "server"
                self.fl_round = config["fl_round"]
                self.crypto.set_client_integer(integer=config["client_integer"])
                self.crypto.set_number_of_bobs(integer=config["number_of_bobs"])
                self.crypto.set_reconstruction_threshold(new_threshold=config["shamir_reconstruction_threshold"])
                self.crypto.set_arithmetic_modulus(modulus=config["arithmetic_modulus"])

                public_keys = self.crypto.generate_public_keys()

                response_dict = {
                    # meta data used for safe checking
                    "sender": f"client",
                    "fl_round": self.fl_round,
                    "client_integer": self.crypto.client_integer,
                    "event_name": Event.ADVERTISE_KEYS.value,
                    # main data
                    "public_encryption_key": public_keys.encryption_key,
                    "public_mask_key": public_keys.mask_key,
                }

            case Event.SHARE_KEYS.value:
                unload_keys = pickle.loads(config["bobs_public_keys"])
                # remove alice
                unload_keys.pop(self.crypto.client_integer)

                self.debugger(f"client integer >>> {self.crypto.client_integer}")

                # register_bobs_keys() will filter out Alice's own key
                self.crypto.register_bobs_keys(bobs_keys_dict=unload_keys)
                self.debugger(6 * "\n", self.crypto.agreed_mask_keys)

            case Event.MASKED_INPUT_COLLECTION.value:
                # not needed assuming no drop out
                pass

            case Event.UNMASKING.value:
                # not needed assuming no drop out (hence no self masking)
                pass
            case _:
                response_dict = {
                    "num_train_samples": self.num_train_samples,
                    "num_val_samples": self.num_val_samples,
                    "response": "client served default",
                }

        return response_dict

    def debugger(self, *info):
        log(DEBUG, 6 * "\n")
        for item in info:
            log(DEBUG, item)

    def _generate_public_keys_dict(self):
        keys = self.crypto.generate_public_keys()

        package = {
            # meta data
            "event_name": Event.ADVERTISE_KEYS.value,
            "fl_round": self.fl_round,
            "client_integer": self.client_integer,
            # important data
            "encryption_key": keys.encryption_key,
            "mask_key": keys.mask_key,
        }

        return package
