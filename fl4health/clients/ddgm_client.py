from logging import DEBUG, INFO, WARN
from pathlib import Path
from collections.abc import Sequence
import time
import torch
import torch.utils
from flwr.common.logger import log
from flwr.common.typing import Config, NDArrays, Scalar
import torch.utils.data

from fl4health.clients.basic_client import BasicClient
from fl4health.parameter_exchange.secure_aggregation_exchanger import SecureAggregationExchanger
from fl4health.utils.losses import LossMeterType
from fl4health.utils.metrics import Metric
from fl4health.checkpointing.client_module import ClientCheckpointAndStateModule

from fl4health.privacy_mechanisms.slow_discrete_gaussian_mechanism import (
    pad_zeros,
    randomized_rounding,
    calculate_l2_upper_bound,
    clip_vector
)
from fl4health.privacy_mechanisms.discrete_gaussian_mechanism import (
    generate_discrete_gaussian_vector,
    fwht
)
from fl4health.privacy_mechanisms.index import PrivacyMechanismIndex
from fl4health.servers.secure_aggregation_utils import get_model_norm, vectorize_model, get_model_layer_types, change_model_dtypes

from fl4health.reporting.base_reporter import BaseReporter

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
        client_number: int | None = None,
        privacy_settings: dict[str, Scalar] = {},
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
        
        self.client_number: int = client_number

        self.privacy_settings = privacy_settings
        self.clipping_bound: float = privacy_settings["clipping_bound"]
        self.noise_multiplier: float = privacy_settings["noise_multiplier"]
        self.dataset_name: str = privacy_settings["dataset"]
                
        self.parameter_exchanger = SecureAggregationExchanger()
        
        self.model_dim = None
        self.padded_model_dim = None
    
        assert 0 <= self.privacy_settings["bias"] < 1, "Bias must be in [0, 1)"
        
        log(INFO, f"Client {self.client_name} initialized")
    
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

        log(INFO, f'finished basic client setup')

        self.init_parameters_vectorized = vectorize_model(self.model).to(self.device)
        
        self.set_sign_vector()

    def set_parameters(self, parameters: NDArrays, config: Config, fitting_round: bool) -> None:
        super().set_parameters(parameters, config, fitting_round)

        # store the parameters get from the server
        self.server_model_this_round_vectorized = vectorize_model(self.model).to(self.device)
    
    def update_after_train(self, local_steps: int, loss_dict: dict[str, float], config: Config) -> None:
        """
        Use this hook method to perform client procedure.
        """
        
        # get parameters after training and concatenate the parameters into a vectorized torch.Tensor
        parameters = vectorize_model(self.model).to(self.device)

        log(INFO, f'Size of parameters: {len(parameters)}, Type of parameters: {parameters.type}')
        
        # delta_parameters = parameters - self.server_model_this_round_vectorized
        
        # pad the delta parameters
        log(INFO, f'len of delta parameters: {len(parameters)}')
        padded_delta_parameters = pad_zeros(parameters)
        padded_model_dim = len(padded_delta_parameters)

        log(INFO, f'Size of padded parameters: {len(padded_delta_parameters)}')
        
        # clip the vector as part of the steps to achieve client-level DP. Note that, we are clipping the model vector
        c, g = self.clipping_bound, self.privacy_settings['granularity']
        clipped_vector = clip_vector(vector=padded_delta_parameters, clip=c, granularity=g)
        
        # hadamard product
        vector = torch.mul(clipped_vector, self.sign_vector.to(clipped_vector.device))
        log(INFO, f'Starting Welsh Hadamard Transform')
        t0 = time.perf_counter()
        vector = fwht(vector)
        t1 = time.perf_counter()
        log(INFO, f'Welsh Hadamard Transform finished in {t1-t0} sec')
        
        # randomized rounding to discrete space
        b = self.privacy_settings['bias']
        l2_upper_bound = calculate_l2_upper_bound(c, g, padded_model_dim, b)
        vector = randomized_rounding(vector, l2_upper_bound)
        vector = vector.cpu().numpy()

        # adding noise if DP is enabled
        if self.privacy_settings['enable_dp'] is True:

            log(INFO, f'Enable DP: adding DP noise')

            v = (self.noise_multiplier / g) ** 2
            log(INFO, f'noise scale: {v}')

            noise_vector = generate_discrete_gaussian_vector(dim=padded_model_dim, variance=v)
            vector += noise_vector

            log(INFO, f'Finished noise calibration.')

        vector = vector % config["arithmetic_modulus"]
        self.ddgm_returned_vector = vector
        log(INFO, f'End of this client step.')
        
    def fit(self, parameters: NDArrays, config: Config) -> tuple[NDArrays, int, dict[str, Scalar]]:
        """
        Normal fit + DDGM client procedure at the end of the round
        Each client returns *a* vector, wrapped into NDArrays, that is the vectorized client side model. 
        """
        log(INFO, f"Client {self.client_name} starting fit")
        
        _, num_train_samples,metrics = super().fit(parameters, config)

        return (
            [self.ddgm_returned_vector],
            num_train_samples,
            metrics,
        ) 