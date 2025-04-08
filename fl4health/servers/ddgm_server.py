# ddgm non-private server
import datetime
import pickle
import timeit
from dataclasses import dataclass
from itertools import product
from logging import DEBUG, INFO, WARN
from typing import Any, Dict, List, Optional, Set, Tuple
from numba import jit, prange
import numpy as np
import torch
from flwr.common import GetPropertiesIns, Parameters, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.common.logger import log
from flwr.common.typing import NDArrays, Scalar
from flwr.server.client_manager import ClientManager, ClientProxy
from flwr.server.history import History
from flwr.server.server import FitResultsAndFailures, fit_clients
from flwr.common.typing import Config
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

torch.set_default_device('cuda' if torch.cuda.is_available() else 'cpu')


class DDGMServer(FlServer):
    def __init__(
        self,
        client_manager: ClientManager,
        strategy: DDGMStrategy,
        fl_config: Config,
        model: Module,
        parameter_exchanger: ExchangerType,
        privacy_settings,
        wandb_reporter: Optional[WandBReporter] = None,
        checkpointer: Optional[TorchModuleCheckpointer] = None,
        timeout: Optional[float] = 30,
        dropout_mode=False,
        task_name: str = '',
        sign_vector: torch.Tensor = None,
        arithmetic_modulus: int = None,
        ) -> None:
        
        self.debug_mode = True
        log(INFO, 'secure aggregation server initializing...')
        assert isinstance(strategy, DDGMStrategy)
        
        checkpoint_and_state_module = BaseServerCheckpointAndStateModule(model=model, parameter_exchanger=parameter_exchanger, model_checkpointers=[checkpointer])
        
        super().__init__(client_manager, fl_config, strategy, wandb_reporter, checkpoint_and_state_module,accept_failures=False)
        
        self.sign_vector = sign_vector
        self.arithmetic_modulus = get_arithmetic_modulus(
            num_of_clients=self._client_manager.num_available(),
            model_integer_range=privacy_settings['model_integer_range'],
        ) if arithmetic_modulus is None else arithmetic_modulus
        
        self.privacy_settings = privacy_settings
        self.accountant: DDGaussAccountant
        
        self.wandb_reporter = wandb_reporter

        self.layer_dtypes = get_model_layer_types(model)

        self.timeout = timeout
        self.dropout_mode = dropout_mode
        self.task_name = task_name
        
        temporary_dir = os.path.join(
            os.path.dirname(checkpointer.checkpoint_path),
            'temp'
        )

        if not os.path.exists(temporary_dir):
            os.makedirs(temporary_dir)

        self.temporary_model_path = os.path.join(
            temporary_dir,
            f'server_initial_model.pth'
        )

        self.blackbox = BlackBox()
        metrics_dir = os.path.join(os.path.dirname(
            checkpointer.checkpoint_path), 
            'metrics'
        )

        if not os.path.exists(metrics_dir):
            os.makedirs(metrics_dir)

        self.metrics_path = os.path.join(
            metrics_dir,
            'server_metrics.json'
        )
        
    # def fit(self, num_rounds: int, timeout: float | None) -> tuple[History, float]:
        
    #     state_load_success = self._load_server_state()
    #     if state_load_success:
    #         log(INFO, "Server state checkpoint successfully loaded.")
    #     else:
    #         log(INFO, "Initializing server state and global parameters")
    #         self.parameters = self._get_initial_parameters(server_round=0, timeout=timeout)
    #         self.history = History()
    #         self.current_round = 1

    #     if self.current_round == 1:
    #         log(INFO, "Evaluating initial parameters")
    #         res = self.strategy.evaluate(0, parameters=self.parameters)
    #         if res is not None:
    #             log(
    #                 INFO,
    #                 "initial parameters (loss, other metrics): %s, %s",
    #                 res[0],
    #                 res[1],
    #             )
    #             self.history.add_loss_centralized(server_round=0, loss=res[0])
    #             self.history.add_metrics_centralized(server_round=0, metrics=res[1])

    #         # Run federated learning for num_rounds
    #         log(INFO, "FL starting")

    #     start_time = datetime.datetime.now()
        
    #     while self.current_round < (num_rounds + 1):
    #         # Train model and replace previous global model
    #         res_fit = self.fit_round(server_round=self.current_round, timeout=timeout)
    #         if res_fit:
    #             parameters_prime, fit_metrics, _ = res_fit  # fit_metrics_aggregated
    #             if parameters_prime:
    #                 self.parameters = parameters_prime
    #             self.history.add_metrics_distributed_fit(server_round=self.current_round, metrics=fit_metrics)

    #         # Evaluate model using strategy implementation
    #         res_cen = self.strategy.evaluate(self.current_round, parameters=self.parameters)
    #         if res_cen is not None:
    #             loss_cen, metrics_cen = res_cen
    #             log(
    #                 INFO,
    #                 "fit progress: (%s, %s, %s, %s)",
    #                 self.current_round,
    #                 loss_cen,
    #                 metrics_cen,
    #                 (datetime.datetime.now() - start_time).total_seconds(),
    #             )
    #             self.history.add_loss_centralized(server_round=self.current_round, loss=loss_cen)
    #             self.history.add_metrics_centralized(server_round=self.current_round, metrics=metrics_cen)

    #         # Evaluate model on a sample of available clients
    #         res_fed = self.evaluate_round(server_round=self.current_round, timeout=timeout)
    #         if res_fed:
    #             loss_fed, evaluate_metrics_fed, _ = res_fed
    #             if loss_fed:
    #                 self.history.add_loss_distributed(server_round=self.current_round, loss=loss_fed)
    #                 self.history.add_metrics_distributed(server_round=self.current_round, metrics=evaluate_metrics_fed)

    #         self.current_round += 1

    #         # Save checkpoint after training and testing
    #         self._save_server_state()

    #     # Bookkeeping
    #     end_time = datetime.datetime.now()
    #     elapsed_time = end_time - start_time
    #     log(INFO, "FL finished in %s", str(elapsed_time))
    #     return self.history, elapsed_time.total_seconds()
    
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
            # sign_vector=self.sign_vector,
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
            Optional[Parameters],
            dict[str, Scalar],
        ] = self.strategy.aggregate_fit(server_round, results, failures,arithmetic_modulus=self.arithmetic_modulus, sign_vector=self.sign_vector)

        parameters_aggregated, metrics_aggregated = aggregated_result
        
        round_end = datetime.datetime.now()
        
        fit_round_results = parameters_aggregated, metrics_aggregated, (results, failures)
        
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