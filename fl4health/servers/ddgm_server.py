# ddgm non-private server
import datetime
import pickle
import timeit
from dataclasses import dataclass
from itertools import product
from logging import DEBUG, INFO, WARN
# from typing import Any, Dict, List, Optional, Set, Tuple
from flwr.common.typing import Code, Config, GetParametersIns, Scalar

from numba import jit, prange
import numpy as np
import torch
from flwr.common import GetPropertiesIns, Parameters, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.common.logger import log
from flwr.common.typing import NDArrays, Scalar
from flwr.server.client_manager import ClientManager, ClientProxy
from flwr.server.history import History
from flwr.server.server import FitResultsAndFailures, fit_clients
from torch.nn import Module

from fl4health.checkpointing.checkpointer import TorchModuleCheckpointer
from fl4health.client_managers.base_sampling_manager import BaseFractionSamplingManager
from fl4health.privacy_mechanisms.slow_discrete_gaussian_mechanism import (
    generate_random_sign_vector,
    generate_walsh_hadamard_matrix,
    get_exponent
)
from fl4health.privacy_mechanisms.discrete_gaussian_mechanism import (
    fwht,
    shift_transform, 
    modular_clipping
)
from fl4health.privacy_mechanisms.index import PrivacyMechanismIndex
from fl4health.reporting.wandb_reporter import WandBReporter
from fl4health.reporting.secure_aggregation_blackbox import BlackBox

from fl4health.parameter_exchange.parameter_exchanger_base import ExchangerType
from fl4health.servers.base_server import FlServer
from fl4health.servers.polling import poll_clients
from fl4health.servers.secure_aggregation_utils import (
    get_model_dimension,
    unvectorize_model,
    vectorize_model,
    get_model_layer_types,
    get_arithmetic_modulus
)
from fl4health.strategies.ddgm_strategy import DDGMStrategy

from fl4health.parameter_exchange.secure_aggregation_exchanger import SecureAggregationExchanger
from fl4health.checkpointing.server_module import BaseServerCheckpointAndStateModule

import json
import os

from fl4health.privacy_mechanisms.gaussian_mechanism import gaussian_mechanism

from fl4health.privacy.distributed_discrete_gaussian_accountant import DDGaussAccountant

torch.set_default_device('cuda' if torch.cuda.is_available() else 'cpu')


class DDGMServer(FlServer):
    def __init__(
        self,
        client_manager: ClientManager,
        strategy: DDGMStrategy,
        fl_config: Config,
        privacy_settings: dict[str, Scalar] | None,
        checkpoint_and_state_module: BaseServerCheckpointAndStateModule | None = None,
        wandb_reporter: WandBReporter | None = None,
        timeout: float | None = 30,
        dropout_mode=False,
        task_name: str = '',
        sign_vector: torch.Tensor = None,
        arithmetic_modulus: int = None,
        ) -> None:
        
        self.debug_mode = True
        log(INFO, 'secure aggregation server initializing...')
        assert isinstance(strategy, DDGMStrategy)
        
        super().__init__(client_manager, fl_config, strategy, wandb_reporter, checkpoint_and_state_module,accept_failures=False)

        log(INFO, f'checkpointer exists: {self.checkpoint_and_state_module.state_checkpointer is not None}')
        
        self.sign_vector = sign_vector
        self.arithmetic_modulus = get_arithmetic_modulus(
            num_of_clients=self._client_manager.num_available(),
            model_integer_range=privacy_settings['model_integer_range'],
        ) if arithmetic_modulus is None else arithmetic_modulus
        
        self.privacy_settings = privacy_settings
        self.accountant: DDGaussAccountant
        
        self.wandb_reporter = wandb_reporter

        self.server_model =  checkpoint_and_state_module.model

        self.timeout = timeout
        self.dropout_mode = dropout_mode
        self.task_name = task_name
        
    
    def fit_round(
        self,
        server_round: int,
        timeout: float | None,
    ) -> tuple[Parameters | None, dict[str, Scalar], FitResultsAndFailures] | None:
        """
        This function is called at each round of federated training. The flow is generally the same as a flower
        server, where clients are sampled and client side training is requested from the clients that are chosen.
        This function simply adds a bit of logging, post processing of the results

        Args:
            server_round (int): Current round number of the FL training. Begins at 1
            timeout (float | None): Time that the server should wait (in seconds) for responses from the clients.
                Defaults to None, which indicates indefinite timeout.

        Returns:
            tuple[Parameters | None, dict[str, Scalar], FitResultsAndFailures] | None: The results of training
                on the client sit. The first set of parameters are the AGGREGATED parameters from the strategy. The
                second is a dictionary of AGGREGATED metrics. The third component holds the individual (non-aggregated)
                parameters, loss, and metrics for successful and unsuccessful client-side training.
        """
        
        """Perform a single round of federated averaging."""
        
        round_start = datetime.datetime.now()
        
        # Get clients and their respective instructions from strategy
        client_instructions = self.strategy.configure_fit(
            server_round=server_round,
            parameters=self.parameters,
            client_manager=self._client_manager,
            arithmetic_modulus=self.arithmetic_modulus,
        )

        if not client_instructions:
            log(INFO, "configure_fit: no clients selected, cancel")
            return None
        log(
            INFO,
            "configure_fit: strategy sampled %s clients (out of %s)",
            len(client_instructions),
            self._client_manager.num_available(),
        )

        # Collect `fit` results from all clients participating in this round
        results, failures = fit_clients(
            client_instructions=client_instructions,
            max_workers=self.max_workers,
            timeout=timeout,
            group_id=server_round,
        )
        log(
            INFO,
            "aggregate_fit: received %s results and %s failures",
            len(results),
            len(failures),
        )

        # Aggregate training results
        aggregated_result: tuple[
            Parameters | None,
            dict[str, Scalar],
        ] = self.strategy.aggregate_fit(server_round, results, failures,arithmetic_modulus=self.arithmetic_modulus, sign_vector=self.sign_vector, server_model=self.server_model)

        parameters_aggregated, metrics_aggregated = aggregated_result
        fit_round_results = parameters_aggregated, metrics_aggregated, (results, failures)

        round_end = datetime.datetime.now()
                
        self.reports_manager.report(
            {
                "fit_round_start": str(round_start),
                "fit_round_end": str(round_end),
                "fit_round_time_elapsed": round((round_end - round_start).total_seconds()),
            },
            server_round,
        )
        if fit_round_results is not None:
            _, metrics, fit_results_and_failures = fit_round_results
            self.reports_manager.report({"fit_round_metrics": metrics}, server_round)
            failures = fit_results_and_failures[1] if fit_results_and_failures else None

            if failures and not self.accept_failures:
                self._log_client_failures(failures)
                self._terminate_after_unacceptable_failures(timeout)


        return fit_round_results
    
    # def get_epsilon_delta(self) -> tuple[float, float]:
        
    #     self.accountant = DDGaussAccountant(
    #         l2_norm_clip=self.privacy_settings['clipping_bound'],
    #         )