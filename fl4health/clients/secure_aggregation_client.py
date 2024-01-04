import pickle
from logging import DEBUG, INFO, WARN
from pathlib import Path
from random import random
from typing import Dict, Optional, Sequence, Tuple

import torch
from flwr.common.logger import log
from flwr.common.typing import Config, List, NDArrays, Scalar

from fl4health.checkpointing.checkpointer import TorchCheckpointer
from fl4health.clients.basic_client import BasicClient
from fl4health.parameter_exchange.parameter_exchanger_base import ParameterExchanger
from fl4health.parameter_exchange.secure_aggregation_exchanger import SecureAggregationExchanger
from fl4health.security.secure_aggregation import ClientCryptoKit, Event, ShamirSecrets
from fl4health.utils.losses import LossMeterType
from fl4health.utils.metrics import Metric, MetricMeterType


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

        # client-side cryptography for Secure Aggregation
        self.crypto = ClientCryptoKit()

        self.parameter_exchanger = SecureAggregationExchanger()
        log(INFO, f"Client initializes parameter exchange as {type(self.parameter_exchanger)}")

    # The 'main' function for client-side secure aggregation.
    def get_properties(self, config: Config) -> Dict[str, Scalar]:
        """Receiver of server calls for Secure Aggregation Protocol."""

        if not self.initialized:
            self.setup_client(config)

        response_dict = {}
        match config["event_name"]:

            case Event.ADVERTISE_KEYS.value:

                # NOTE this client integer ID currently persists across SecAgg rounds
                self.crypto.set_client_integer(integer=config["client_integer"])

                # these determine the number of Shamir shares
                self.crypto.set_number_of_bobs(integer=config["number_of_bobs"])
                self.crypto.set_reconstruction_threshold(new_threshold=config["shamir_reconstruction_threshold"])

                # modulus may change at the start of each SecAgg if dropout occurs (refer to documentation)
                self.crypto.set_arithmetic_modulus(modulus=config["arithmetic_modulus"])

                public_keys = self.crypto.generate_public_keys()

                response_dict = {
                    # main data
                    "client_integer": self.crypto.client_integer,
                    "public_encryption_key": public_keys.encryption_key,
                    "public_mask_key": public_keys.mask_key,
                    # for server-side validation
                    "event_name": Event.ADVERTISE_KEYS.value,
                }

            case Event.SHARE_KEYS.value:

                unload_keys = pickle.loads(config["bobs_public_keys"])
                unload_keys.pop(self.crypto.client_integer)  # remove Alice herself

                t = self.crypto.reconstruction_threshold
                if len(unload_keys) < t:

                    error_msg = f"""
                    Too many droped out clients (#peers){len(unload_keys)} < threshold {t},
                    aborting client {self.crypto.client_integer}.
                    """

                    log(WARN, error_msg)

                    # NOTE open problem | when one client triggers everyone to abort FL,
                    # how does server know it's not malicious?
                    exit()

                # key agreement and storage
                self.crypto.register_bobs_keys(bobs_keys_dict=unload_keys)

                # ****************** debug point (starts) *******************

                # log(DEBUG, self.crypto.agreed_mask_keys)
                # log(DEBUG, self.crypto.agreed_encryption_keys)

                # ****************** debug point (ends) *********************

                # generate self-mask seed
                self.crypto.set_self_mask_seed()

                # Shamir shares for 1) self-mask and 2) pairmask secrete key
                shamir_pair_self = self.crypto.get_encrypted_shamir_shares()

                response_dict = {
                    "event_name": Event.SHARE_KEYS.value,
                    "client_integer": self.crypto.client_integer,
                    "serialized_encrypted_shamir": pickle.dumps(shamir_pair_self),
                }

            case Event.MASKED_INPUT_COLLECTION.value:

                self.crypto.bob_shamir_secrets = {}
                received = pickle.loads(config["pickled_message"])  # expects dict

                t = self.crypto.reconstruction_threshold
                if len(received) < t:

                    error_msg = f"""
                    Too many droped out clients (#peers){len(received)} < threshold {t},
                    aborting client {self.crypto.client_integer}.
                    """

                    log(WARN, error_msg)

                    # NOTE triggers everyone to abort FL,
                    exit()

                # receive Shamir shares from other clients
                self.crypto.register_shamir_shares(shamir_shares=received)

                response_dict = {
                    "event_name": Event.MASKED_INPUT_COLLECTION.value,
                    "client_integer": self.crypto.client_integer,
                }

            case Event.UNMASKING.value:
                # not needed assuming no drop out (hence no self masking)
                dropout_status = pickle.loads(config["dropout_status"])

                shamir_secrets = {}
                for id, has_dropped in dropout_status:
                    secret: ShamirSecrets = self.crypto.bob_shamir_secrets[id]
                    shamir_secrets[id] = secret.pairwise if has_dropped else secret.individual

                response_dict = {**response_dict, "shamir_secrets": pickle.dumps(shamir_secrets)}
            case _:
                response_dict = {
                    "num_train_samples": self.num_train_samples,
                    "num_val_samples": self.num_val_samples,
                    "response": "client served default",
                }

        return response_dict

    # Orchestrates training
    def fit(self, parameters: NDArrays, config: Config) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        local_epochs, local_steps, current_server_round = self.process_config(config)
        if not self.initialized:
            self.setup_client(config)
        self.set_parameters(parameters, config)

        if current_server_round == 0:
            self.constant_parameters(n=0)
            # NOTE this is a full exchanger, needs to be modified for any partial exchanger
            parmeters = [val.cpu().numpy() for _, val in self.model.state_dict().items()]
            metrics = {}
            return (
                parmeters,
                self.num_train_samples,
                metrics,
            )
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
        # self.modify_parmeters()

        # pickle.dump(self.model.state_dict(), open(f"examples/secure_aggregation_example/{random()}.pkl", "wb"))
        # self.constant_parameters(n=0)
        # self.debugger(self.model.state_dict().values())
        # self.constant_parameters(n=8)
        return (
            self.get_parameters(config),
            self.num_train_samples,
            metrics,
        )

    def get_parameter_exchanger(self, config: Config) -> ParameterExchanger:
        """
        Returns Full Parameter Exchangers. Subclasses that require custom Parameter Exchangers can override this.
        """
        return SecureAggregationExchanger()

    def get_parameters(self, config: Config) -> NDArrays:
        """
        Determines which weights are sent back to the server for aggregation. This uses a parameter exchanger to
        determine parameters sent
        Args:
            config (Config): The config is sent by the FL server to allow for customization in the function if desired

        Returns:
            NDArrays: These are the parameters to be sent to the server. At minimum they represent the relevant model
                parameters to be aggregated, but can contain more information
        """

        assert self.model is not None and self.parameter_exchanger is not None

        self.parameter_exchanger = SecureAggregationExchanger()

        # TODO
        self.debugger(f"Parameter exchanger type at get_parameters: {type(self.parameter_exchanger)}")
        assert isinstance(self.parameter_exchanger, SecureAggregationExchanger)
        v = self.generate_mask()
        self.debugger("mask vector=====", v[:5])
        return self.parameter_exchanger.push_parameters(model=self.model, mask=v, config=config)

    def generate_mask(self):
        dim = sum(param.numel() for param in self.model.parameters() if param.requires_grad)
        # computes masking vector; this can only be run after masking seed agreement
        # modify to add self masking
        self.debugger(f"client integer {self.crypto.client_integer}", self.crypto.agreed_mask_keys)
        pair_mask_vect: List[int] = self.crypto.get_pair_mask_sum(vector_dim=dim)
        return pair_mask_vect

    def constant_parameters(self, n=0) -> None:
        """Sets all parms to constant. For testing and pre-training init (fl_round=0) only."""
        zero_param_dict = self.model.state_dict()
        for layer_name, params in zero_param_dict.items():
            zero_param_dict[layer_name] = n * torch.ones(params.shape).to(torch.int64)
            # self.debugger(f'round {self.fl_round} >>>>', zero_param_dict[layer_name].dtype, zero_param_dict[layer_name].size())
        self.model.load_state_dict(zero_param_dict)

    def modify_parmeters(self):
        "Use this method to post process model parameters (i.e. masking & noising) after train and distributed evaluation."

        # counts trainable model parameters
        dim = sum(param.numel() for param in self.model.parameters() if param.requires_grad)

        # computes masking vector; this can only be run after masking seed agreement
        pair_mask_vect: List[int] = self.crypto.get_pair_mask_sum(
            vector_dim=dim
        )  # pass in online_clients kwarg for drop out case
        # self.debugger('masking vector', pair_mask_vect)
        # modify parms  << quantization for privacy mechanism + modular arithmetic>>
        # pair_mask_vect = SkellamMechanism(query_vector=pair_mask_vect, skellam_variance=10)
        # pair_mask_vect = list(range(dim)) # TODO testing only
        i = 0
        params_dict = self.model.state_dict()
        for name, params in params_dict.items():
            j = i + params.numel()
            mask = torch.tensor(pair_mask_vect[i:j], dtype=torch.float64).reshape(params.size())
            params_dict[name] = 0 * torch.ones(
                params.shape, dtype=torch.float64
            )  # TEST ONLY: remove zeroing of params
            params_dict[name] += mask
            i = j

        # load modified parms
        self.model.to(torch.float64)
        self.model.load_state_dict(params_dict)
        # self.debugger(self.model.state_dict().values())

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

    def get_metadata(self, event_name: Optional[str] = None) -> Dict[str, Scalar]:
        metadata = {
            "sender": "client",
            "client_integer": self.crypto.client_integer,
            "sec_agg_round": self.fl,
        }
        if event_name:
            metadata["event_name"] = event_name
        return metadata
