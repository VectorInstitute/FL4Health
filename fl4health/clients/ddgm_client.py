import pickle
from logging import DEBUG, INFO, WARN
from pathlib import Path
import random
import copy
from typing import Dict, Optional, Sequence, Tuple
import time
import torch
import numpy
from opacus.data_loader import DPDataLoader
import torch.utils
import gc
from torch.utils.data import Subset, DataLoader
from flwr.common.logger import log
from flwr.common.typing import Config, List, NDArrays, Scalar
import torch.utils.data

from fl4health.checkpointing.checkpointer import TorchModuleCheckpointer
from fl4health.clients.basic_client import BasicClient
from fl4health.parameter_exchange.parameter_exchanger_base import ParameterExchanger
from fl4health.parameter_exchange.secure_aggregation_exchanger import SecureAggregationExchanger
from fl4health.privacy_mechanisms.index import PrivacyMechanismIndex
from fl4health.security.secure_aggregation import ClientCryptoKit, ClientId, Event, ShamirSecret, ShamirSecrets
from fl4health.utils.losses import LossMeterType
from fl4health.utils.metrics import Metric
from fl4health.checkpointing.client_module import CheckpointMode, ClientCheckpointAndStateModule


from fl4health.privacy_mechanisms.slow_discrete_gaussian_mechanism import (
    discrete_gaussian_noise_vector,
    generate_random_sign_vector,
    generate_walsh_hadamard_matrix,
    pad_zeros,
    randomized_rounding,
    calculate_delta_squared,
    calculate_tau,
    single_fl_round_concentrated_dp,
    clip_vector,
    get_exponent
)
from fl4health.privacy_mechanisms.discrete_gaussian_mechanism import (
    generate_discrete_gaussian_vector,
    fwht,
    shift_transform
)
from fl4health.privacy_mechanisms.index import PrivacyMechanismIndex
from fl4health.servers.secure_aggregation_utils import get_model_norm, vectorize_model, get_model_layer_types, change_model_dtypes
import json 
import os
import uuid 
import timeit

from fl4health.privacy_mechanisms.gaussian_mechanism import gaussian_mechanism
from statistics import mean
from opacus import PrivacyEngine

from fl4health.reporting.base_reporter import BaseReporter

# torch.cuda.init()

torch.set_default_device('cuda' if torch.cuda.is_available() else 'cpu')
# torch.set_default_dtype(torch.float64)

from fl4health.utils.config import narrow_dict_type

class DDGMClient(BasicClient):
    def __init__(
        self,
        data_path: Path,
        metrics: Sequence[Metric],
        device: torch.device,
        loss_meter_type: LossMeterType = LossMeterType.AVERAGE,
        checkpoint_and_state_module: ClientCheckpointAndStateModule | None = None,
        reporters: Sequence[BaseReporter] | None = None,
        progress_bar: bool = False,
        client_name: str | None = None,
        privacy_settings: Dict[str, Scalar] = {},
        # sign_vector: torch.Tensor = None,
    ) -> None:
        
        super().__init__(
            data_path=data_path,
            metrics=metrics,
            device=device,
            loss_meter_type=loss_meter_type,
            checkpoint_and_state_module=checkpoint_and_state_module,
            reporters=reporters,
            progress_bar=progress_bar,
            client_name=client_name,
        )
        
        # self._set_path(self.client_name)
        
        self.privacy_settings = privacy_settings
        self.clipping_bound: float = privacy_settings["clipping_bound"]
        self.noise_multiplier: float = privacy_settings["noise_multiplier"]
        
        # self.sign_vector = sign_vector
        
        self.parameter_exchanger = SecureAggregationExchanger()
        
        self.model_dim = None
        self.padded_model_dim = None
    
        assert 0 <= self.privacy_settings["bias"] < 1, "Bias must be in [0, 1)"
        
        log(INFO, f"Client {self.client_name} initialized")
    
        
    # def _set_path(self, client_name) -> None:
    #     """
    #     Set the path for the client's temporary model and secrets
    #     """
    #     temporary_dir = os.path.join(
    #         os.path.dirname(checkpointer.checkpoint_path),
    #         'temp'
    #     )
    #     self.temporary_dir = temporary_dir

    #     if not os.path.exists(temporary_dir):
    #         os.makedirs(temporary_dir)

    #     # path for model vector 
    #     self.temporary_model_path = os.path.join(
    #         temporary_dir,
    #         f'client_{self.client_id}_initial_model.pth'
    #     )

    #     # path to torch.save() model
    #     # NOTE saving state dict can help avoid Opacus errors.
    #     self.temporary_model_state_path = os.path.join(
    #         temporary_dir,
    #         f'client_{self.client_id}_initial_model_state.pth'
    #     )
    #     self.temporary_optimizer_state_path = os.path.join(
    #         temporary_dir,
    #         f'client_{self.client_id}_optimizer_state.pth'
    #     )

    #     metrics_dir = os.path.join(
    #         os.path.dirname(checkpointer.checkpoint_path),
    #         'metrics'
    #     )

    #     if not os.path.exists(metrics_dir):
    #         os.makedirs(metrics_dir)

    #     self.metrics_path = os.path.join(
    #         metrics_dir,
    #         f'client_{self.client_id}_metrics.json'
    #     )
    
    def set_sign_vector(self) -> None:
        """
        Hook method to set the sign vector.
        """
        pass
        
    def setup_client(self, config: Config) -> None:
        # Ensure that clipping bound and noise multiplier is present in config
        # Set attributes to be used when setting DP training

        # Do basic client setup
        super().setup_client(config) 
        
        # set local steps as 1
        self.local_epochs = None
        self.local_steps = 1
        
        self.set_sign_vector()
        
    # def set_parameters(self, parameters: NDArrays, config: Config, fitting_round: bool) -> None:
    #     "Store the parameters from the server"
    #     self.init_parameters = torch.tensor(parameters)
        
    #     super().set_parameters(parameters, config, fitting_round)
    
    def update_after_train(self, local_steps: int, loss_dict: dict[str, float], config: Config) -> None:
        """
        Use this hook method to perform client procedure.
        """
        
        parameters = torch.from_numpy(self.parameter_exchanger.push_parameters(self.model, config=config))
        
        padded_model_dim = 2**get_exponent(len(parameters))
        
        delta_parameters = parameters - self.init_parameters
        
        # pad the delta parameters
        padded_delta_parameters = pad_zeros(delta_parameters, padded_model_dim)
        
        c, g = self.clipping_bound, self.privacy_settings['granularity']
        
        clipped_vector = clip_vector(vector=delta_parameters, clip=c, granularity=g)
        
        # sign_vector = generate_random_sign_vector(dim=self.padded_model_dim)
        sign_vector = config["sign_vector"]
        
        # hadamard product
        vector = torch.mul(clipped_vector, self.sign_vector)
        log(INFO, f'Starting Welsh Hadamard Transform')
        t0 = time.perf_counter()
        vector = fwht(vector)
        t1 = time.perf_counter()
        log(INFO, f'Welsh Hadamard Transform finished in {t1-t0} sec')
        
        b = self.privacy_settings['bias']
        delta_squared = calculate_delta_squared(c, g, self.model_dim, b, mini_client_size)
        vector = randomized_rounding(vector, delta_squared, g)
        
        v = (self.noise_multiplier / g) ** 2
        
        noise_vector = generate_discrete_gaussian_vector(dim=self.padded_model_dim, variance=v)
        
        vector += noise_vector
        vector = vector % config["arithmetic_modulus"]
        
        self.ddgm_returned_vector = vector.cpu().numpy()
        
        
        
    def fit(self, parameters: NDArrays, config: Config) -> tuple[NDArrays, int, dict[str, Scalar]]:
        """
        Normal fit + DDGM client procedure at the end of the round
        """
        log(INFO, f"Client {self.client_name} starting fit")
        
        _, num_train_samples,metrics = super().fit(parameters, config)
        
        return (
            self.ddgm_returned_vector,
            num_train_samples,
            metrics,
        ) 