from logging import WARNING
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from flwr.common import (
    EvaluateRes,
    FitRes,
    GetPropertiesIns,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from fl4health.client_managers.base_sampling_manager import BaseFractionSamplingManager
from fl4health.parameter_exchange.parameter_packer import PrincipalComponentsPacker
from fl4health.strategies.strategy_with_poll import StrategyWithPolling


class FedPCA(FedAvg, StrategyWithPolling):
    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
    ) -> None:
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        )

        self.principal_components_packer = PrincipalComponentsPacker()

    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
        """Initialize the (global) model parameters.

        FedPCA does not need initial global model parameters, so
        an empty array is used here.
        """
        return Parameters(tensors=[], tensor_type="numpy.ndarray")

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate client principal components."""

        if not results:
            return None, {}
        client_eigenvalues = []
        client_pcs = []
        for _, fit_res in results:
            A = parameters_to_ndarrays(fit_res.parameters)
            pcs, eigenvalues = self.principal_components_packer.unpack(A)
            client_pcs.append(pcs)
            client_eigenvalues.append(eigenvalues)

        eigenvals_diag = [np.diag(evalue_vec) for evalue_vec in client_eigenvalues]
        X = [U.T @ S for U, S in zip(client_pcs, eigenvals_diag)]
        svd_input = np.concatenate(X, axis=1)
        new_pcs, new_eigenvalues, _ = np.linalg.svd(svd_input)
        parameters_aggregated = ndarrays_to_parameters(self.principal_components_packer.pack(new_pcs, new_eigenvalues))
        return parameters_aggregated, {}

    def configure_poll(
        self, server_round: int, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, GetPropertiesIns]]:
        """
        This function configures everything required to request properties from ALL of the clients. The client
        manger, regardless of type, is instructed to grab all available clients to perform the polling process.

        Args:
            server_round (int): Indicates the server round we're currently on.
            client_manager (ClientManager): The manager used to sample all available clients.

        Returns:
            List[Tuple[ClientProxy, GetPropertiesIns]]: List of sampled client identifiers and the configuration
                to be sent to each client (packaged as GetPropertiesIns).
        """
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)

        property_ins = GetPropertiesIns(config)

        if isinstance(client_manager, BaseFractionSamplingManager):
            clients = client_manager.sample_all(min_num_clients=self.min_available_clients)
        else:
            # Grab all available clients using the basic Flower client manager
            num_available_clients = client_manager.num_available()
            clients = client_manager.sample(num_available_clients, min_num_clients=self.min_available_clients)

        # Return client/config pairs
        return [(client, property_ins) for client in clients]

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        metrics_aggregated = {}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No evaluate_metrics_aggregation_fn provided")

        return 0.0, metrics_aggregated
