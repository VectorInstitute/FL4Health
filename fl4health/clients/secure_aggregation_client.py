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
from fl4health.privacy_mechanisms.index import PrivacyMechanismIndex
from fl4health.security.secure_aggregation import ClientCryptoKit, ClientId, Event, ShamirSecret, ShamirSecrets
from fl4health.utils.losses import LossMeterType
from fl4health.utils.metrics import Metric, MetricMeterType

from fl4health.privacy_mechanisms.discrete_gaussian_mechanism import (
    discrete_gaussian_noise_vector,
    generate_random_sign_vector,
    generate_walsh_hadamard_matrix,
    pad_zeros,
    randomized_rounding,
    clip_vector,
    get_exponent
)
from fl4health.privacy_mechanisms.index import PrivacyMechanismIndex
from fl4health.server.secure_aggregation_utils import get_model_norm, vectorize_model


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
        self.dropout_mode = None

        self.parameter_exchanger = SecureAggregationExchanger()
        log(INFO, f"Client initializes parameter exchange as {type(self.parameter_exchanger)}")

        # set after model initialization
        self.model_dim = None
        self.padded_model_dim = None

        # TODO set differential privacy parameters
        self.privacy_settings = {
            "dp_mechanism": PrivacyMechanismIndex.DiscreteGaussian.value,
            "noise_scale": 1,
            "granularity": 1,
            "clipping_threshold": 10,
            "bias": 0.5,
        }

        assert 0 <= self.privacy_settings["bias"] < 1

    # The 'main' function for client-side secure aggregation.
    def get_properties(self, config: Config) -> Dict[str, Scalar]:
        """Receiver of server calls for Secure Aggregation Protocol."""

        if not self.initialized:
            self.setup_client(config)

        match config["event_name"]:

            case Event.ADVERTISE_KEYS.value:

                # NOTE this client integer ID currently persists across SecAgg rounds
                self.crypto.set_client_integer(integer=config["client_integer"])

                # these determine the number of Shamir shares
                self.crypto.set_number_of_bobs(integer=config["number_of_bobs"])
                self.crypto.set_reconstruction_threshold(new_threshold=config["shamir_reconstruction_threshold"])

                # modulus may change at the start of each SecAgg if dropout occurs (refer to documentation)
                self.crypto.set_arithmetic_modulus(modulus=config["arithmetic_modulus"])

                self.dropout_mode = config["dropout_mode"]

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
                # this case is not executed if we assume no dropouts (hence no need for mask removal)

                # NOTE we do not check U4 is a subset of U3
                # since we opt out of Round 3 (Consistency Check) of SecAgg
                self.debugger("oh my good santa")
                pairmask_shamir_secrets: Dict[ClientId, ShamirSecret] = {}
                for id in pickle.loads(config["pickled_dropout_clients"]):
                    # NOTE these secrets should be unencrypted
                    pairmask_shamir_secrets[id] = self.crypto.bob_shamir_secrets[id].pairwise

                selfmask_shamir_secrets: Dict[ClientId, ShamirSecret] = {}
                for id in pickle.loads(config["pickled_online_clients"]):
                    # NOTE these secrets should be unencrypted
                    selfmask_shamir_secrets[id] = self.crypto.bob_shamir_secrets[id].individual

                response_dict = {
                    "event_name": Event.UNMASKING.value,
                    "current_fl_round": config["current_fl_round"],
                    "pickled_pairmask_shamir_secrets": pickle.dumps(pairmask_shamir_secrets),
                    "pickled_selfmask_shamir_secrets": pickle.dumps(selfmask_shamir_secrets),
                }
                self.debugger("oh my good santa end")
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

        # local model <- global model
        self.set_parameters(parameters, config)

        # TODO round 0 is a temporary work around for server to sample clients
        if current_server_round == 0:
            # set all local model parameters to 0
            self.constant_parameters(n=0)

            # set dimension
            self.model_dim = sum(param.numel() for param in self.model.parameters() if param.requires_grad)
            self.padded_model_dim = 2**get_exponent(self.model_dim)

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

        # NOTE uncomment for debugging
        # pickle.dump(self.model.state_dict(), open(f"examples/secure_aggregation_example/local_models/{random()}.pkl", "wb"))

        log(INFO, f'Number of training examples: {self.num_train_samples}')

        if self.privacy_settings["dp_mechanism"] == PrivacyMechanismIndex.DiscreteGaussian.value:
            m = self.crypto.arithmetic_modulus
            assert isinstance(m, int) and m > 1
            self.privacy_settings["arithmetic_modulus"] = m
        else:
            dp_kind = self.privacy_settings["dp_mechanism"]
            log(WARN, f"The DP-mechanism you chose {dp_kind} is not yet implemented. Aborting FL.")
            exit()

        return (
            self.process_model_post_training(weight=self.num_train_samples),
            self.num_train_samples,
            metrics,
        )

    def process_model_post_training(self, weight: int) -> NDArrays:
        """See Algorithm 1 in 
        Ref
            The Distributed Discrete Gaussian Mechanism for Federated Learning with Secure Aggregation
            https://arxiv.org/pdf/2102.06387.pdf"""

        assert self.model is not None and self.parameter_exchanger is not None
        self.parameter_exchanger = SecureAggregationExchanger()

        # reshape model tensors and concatenate into a vector
        vector = vectorize_model(self.model)

        # TODO adjust clip to weighting
        vector *= weight # weight for FedAvg, server divides out sum of client weights

        c, g = self.privacy_settings['clipping_threshold'], self.privacy_settings['granularity']

        # adjust for scaling of model vector by the weight (often the train data size)
        weighted_clip = c * weight
        vector = clip_vector(vector=vector, clip=weighted_clip, granularity=g)

        # x'' = H(Dx')
        vector = torch.mul(vector, generate_random_sign_vector(dim=self.model_dim)) # hadamard product
        vector = pad_zeros(vector=vector, dim=self.model_dim) # pad zeros 
        vector = torch.matmul(
            input=generate_walsh_hadamard_matrix(exponent=get_exponent(self.model_dim)),
            other=vector
        )

        b = self.privacy_settings['bias']
        vector = randomized_rounding(vector=vector, clip=c, granularity=g, unpadded_model_dim=self.model_dim, bias=b)

        # discrete Gaussian noise 
        v = (self.privacy_settings['noise_scale'] / g) ** 2
        vector += discrete_gaussian_noise_vector(d=self.padded_model_dim, variance=v)

        # TODO if dropout is turned on, then add selfmask below
        vector += torch.tensor(self.crypto.get_pair_mask_sum(vector_dim=self.padded_model_dim, allow_dropout=False))

        processed = []
        i = 0
        for layer in self.model.state_dict.values():
            j = i + layer.numel()
            tensor = vector[i: j].reshape(layer.size()) # de-vectorize
            processed.append(tensor.cpu.numpy())
            i = j

        return processed

    def generate_mask(self):
        dim = sum(param.numel() for param in self.model.parameters() if param.requires_grad)
        # computes masking vector; this can only be run after masking seed agreement
        # modify to add self masking
        self.debugger(f"client integer {self.crypto.client_integer}", self.crypto.agreed_mask_keys)

        if self.dropout_mode:
            return self.crypto.get_duo_mask()

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

    def get_parameter_exchanger(self, config: Config) -> ParameterExchanger:
        """
        Returns Full Parameter Exchangers. Subclasses that require custom Parameter Exchangers can override this.
        """
        return SecureAggregationExchanger()
    

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
