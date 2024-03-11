from fl4health.server.base_server import FlServerWithCheckpointing


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
from torch.nn import Module

from fl4health.checkpointing.checkpointer import TorchCheckpointer
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
from fl4health.reporting.fl_wanb import ServerWandBReporter
from fl4health.reporting.secure_aggregation_blackbox import BlackBox
from fl4health.security.secure_aggregation import (
    ClientId,
    DestinationClientId,
    EllipticCurvePrivateKey,
    Event,
    Seed,
    ServerCryptoKit,
    ShamirOwnerId,
)
from fl4health.server.base_server import ExchangerType, FlServerWithCheckpointing
from fl4health.server.polling import poll_clients
from fl4health.server.secure_aggregation_utils import get_model_dimension, unvectorize_model, vectorize_model, change_model_dtypes, get_model_layer_types
from fl4health.strategies.central_dp_strategy import CentralDPStrategy

from fl4health.parameter_exchange.secure_aggregation_exchanger import SecureAggregationExchanger
import math 
import json
import os

from fl4health.privacy_mechanisms.gaussian_mechanism import gaussian_mechanism

torch.set_default_device('cuda' if torch.cuda.is_available() else 'cpu')


class CentralDPServer(FlServerWithCheckpointing):
    """Central-DP with Continuous Gaussian Mechanism."""
    def __init__(
        self,
        *,
        model: Module,
        privacy_settings,
        client_manager: ClientManager,
        parameter_exchanger: ExchangerType,
        strategy: CentralDPStrategy,
        timeout: Optional[float] = 30,
        checkpointer: Optional[TorchCheckpointer] = None,
        wandb_reporter: Optional[ServerWandBReporter] = None,
    ) -> None:
        
        log(INFO, 'Central-DP server initializing...')
        assert isinstance(strategy, CentralDPStrategy)
        super().__init__(client_manager, model, parameter_exchanger, wandb_reporter, strategy, checkpointer)

        self.timeout = timeout
        self.model_dimension = get_model_dimension(model)

        temporary_dir = os.path.join(
            os.path.dirname(checkpointer.best_checkpoint_path),
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
            checkpointer.best_checkpoint_path), 
            'metrics'
        )

        if not os.path.exists(metrics_dir):
            os.makedirs(metrics_dir)

        self.metrics_path = os.path.join(
            metrics_dir,
            'server_metrics.json'
        )

        # differential privacy
        self.privacy_settings = {
            **privacy_settings,
            "dp_mechanism": PrivacyMechanismIndex.ContinuousGaussian.value,
        }

        self.gaussian_noise_variance = self.privacy_settings['gaussian_noise_variance']

        with open(self.metrics_path, 'w+') as file:
            json.dump({
                'privacy_hyperparameters': self.privacy_settings
            }, file)
            
    def fit(self, num_rounds: int, timeout: Optional[float]) -> History:

        history = History()

        # Initialize parameters
        log(INFO, "Initializing global parameters")
        self.parameters = self._get_initial_parameters(timeout=timeout)
        log(INFO, "Evaluating initial parameters")
        res = self.strategy.evaluate(0, parameters=self.parameters)
        if res is not None:
            log(
                INFO,
                "initial parameters (loss, other metrics): %s, %s",
                res[0],
                res[1],
            )
            history.add_loss_centralized(server_round=0, loss=res[0])
            history.add_metrics_centralized(server_round=0, metrics=res[1])

        # Run federated learning for num_rounds
        log(INFO, "FL starting")
        start_time = timeit.default_timer()

        for current_round in range(1, num_rounds + 1):
                    
            inital_model_vector = vectorize_model(self.server_model)
            torch.save(inital_model_vector, self.temporary_model_path)
            del inital_model_vector

            # Train model and replace previous global model
            metrics, statistics = self.dp_fit_round(server_round=current_round, timeout=timeout)

            history.add_metrics_distributed_fit(server_round=current_round, metrics=metrics)

            # Evaluate model using strategy implementation
            res_cen = self.strategy.evaluate(current_round, parameters=self.parameters)
            if res_cen is not None:
                loss_cen, metrics_cen = res_cen
                log(
                    INFO,
                    "fit progress: (%s, %s, %s, %s)",
                    current_round,
                    loss_cen,
                    metrics_cen,
                    timeit.default_timer() - start_time,
                )
                history.add_loss_centralized(server_round=current_round, loss=loss_cen)
                history.add_metrics_centralized(
                    server_round=current_round, metrics=metrics_cen
                )

            # Evaluate model on a sample of available clients
            res_fed = self.evaluate_round(server_round=current_round, timeout=timeout)
            if res_fed:
                loss_fed, evaluate_metrics_fed, _ = res_fed
                if loss_fed:
                    history.add_loss_distributed(
                        server_round=current_round, loss=loss_fed
                    )
                    history.add_metrics_distributed(
                        server_round=current_round, metrics=evaluate_metrics_fed
                    )
                
                metrics_to_save = {}

                with open(self.metrics_path, 'r') as file:
                    metrics_to_save = json.load(file)

                    metrics_to_save['current_round'] = current_round

                    if current_round == 1:
                        metrics_to_save['privacy_hyperparameters']['num_fl_rounds'] = num_rounds
                        metrics_to_save['privacy_hyperparameters']['num_clients'] = statistics[1]

                    for key, value in metrics.items():
                        if key not in metrics_to_save:
                            metrics_to_save[key] = [value]
                        else:
                            metrics_to_save[key].append(value)

                    if 'loss' not in metrics_to_save:
                        metrics_to_save['loss'] = [loss_fed]
                    else:
                        metrics_to_save['loss'].append(loss_fed)

                    for key, value in evaluate_metrics_fed.items():
                        if key not in metrics_to_save:
                            metrics_to_save[key] = [value]
                        else:
                            metrics_to_save[key].append(value)

                    now = timeit.default_timer()
                    if 'time' not in metrics_to_save:
                        metrics_to_save['time'] = [now-start_time]
                    else:
                        metrics_to_save['time'].append(now-start_time)


                with open(self.metrics_path, 'w') as file:
                    json.dump(metrics_to_save, file)
                    log(DEBUG, f'finished recording metrics for round {current_round}')

        # Bookkeeping
        end_time = timeit.default_timer()
        elapsed = end_time - start_time
        log(INFO, "FL finished in %s", elapsed)
        return history
    

    def dp_fit_round(
        self,
        server_round: int,
        timeout: Optional[float],
    ) -> Optional[
        Tuple[Optional[Parameters], Dict[str, Scalar], FitResultsAndFailures]
    ]:
        """Perform a single round of federated averaging."""

        # Get clients and their respective instructions from strategy
        client_instructions = self.strategy.configure_fit(
            server_round=server_round,
            parameters=self.parameters,
            client_manager=self._client_manager,
        )

        if not client_instructions:
            log(INFO, "fit_round %s: no clients selected, cancel", server_round)
            return None
        log(
            DEBUG,
            "fit_round %s: strategy sampled %s clients (out of %s)",
            server_round,
            len(client_instructions),
            self._client_manager.num_available(),
        )

        # Collect `fit` results from all clients participating in this round
        results, failures = fit_clients(
            client_instructions=client_instructions,
            max_workers=self.max_workers,
            timeout=timeout,
        )
        log(
            DEBUG,
            "fit_round %s received %s results and %s failures",
            server_round,
            len(results),
            len(failures),
        )

        # CentralDPStrategy is customize & has different 
        # aggregate_fit output types than BasicFedAvg stragegy.
        assert isinstance(self.strategy, CentralDPStrategy)

        # Aggregate training results
        aggregated_result: Tuple[
            Optional[Parameters],
            Dict[str, Scalar],
        ] = self.strategy.aggregate_fit(server_round, results, failures)

        log(DEBUG, f'aggregted_result')
        log(DEBUG, aggregated_result)

        # statistics is (aggregate_trainset_size, client_count)
        global_model_delta_vector, metrics_aggregated, statistics = aggregated_result
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # model delta
        delta = torch.from_numpy(global_model_delta_vector).to(device=device)
        
        # noisy delta 
        sigma = math.sqrt(self.gaussian_noise_variance)
        delta += gaussian_mechanism(dim=self.model_dimension, standard_deviation=sigma)
        
        model_vector = delta + torch.load(self.temporary_model_path).to(device=device) 
        self.server_model = unvectorize_model(self.server_model, model_vector)
        self.parameters = ndarrays_to_parameters(
            [layer.cpu().numpy() for layer in self.server_model.state_dict().values()]
        )

        return metrics_aggregated, statistics