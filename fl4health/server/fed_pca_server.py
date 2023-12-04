import timeit
from logging import DEBUG, INFO, WARNING
from typing import Dict, List, Optional, Tuple, Union

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.history import History
from flwr.server.server import Server, evaluate_clients, fit_clients
from flwr.server.strategy import Strategy

from fl4health.parameter_exchange.parameter_packer import PrincipalComponentsPacker
from fl4health.PCA.pca import ServerSideMerger

FitResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, FitRes]],
    List[Union[Tuple[ClientProxy, FitRes], BaseException]],
]

EvaluateResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, EvaluateRes]],
    List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
]


class FedPCAServer(Server):
    """
    This server is responsible for orchestrating the execution of federated PCA.

    Args:
        client_manager (ClientManager): Determines the mechanism by which clients are sampled by the server, if
                they are to be sampled at all.
        server_side_merger (ServerSideMerger): responsible for
        merging local clients' principal components.
        strategy (Optional[Strategy], optional): The aggregation strategy to be used by the server to handle.
        For FedPCA, the strategy will not be used.
    """

    def __init__(
        self,
        server_side_merger: ServerSideMerger,
        client_manager: ClientManager,
        strategy: Optional[Strategy],
        min_num_clients: int,
    ) -> None:
        if strategy is not None:
            log(WARNING, "strategy will not be used in federated pca.")

        self._client_manager: ClientManager = client_manager
        self.parameters: Parameters = Parameters(tensors=[], tensor_type="numpy.ndarray")
        self.server_side_merger = server_side_merger
        self.max_workers: Optional[int] = None
        self.principal_components_packer = PrincipalComponentsPacker()
        self.min_num_clients = min_num_clients

    def fit(self, num_rounds: int, timeout: Optional[float]) -> History:
        """Run federated PCA."""
        history = History()
        # Run federated PCA
        log(INFO, "Federated PCA starting")
        start_time = timeit.default_timer()
        for current_round in range(1, num_rounds + 1):
            # Train model and replace previous global model
            self.fit_round(server_round=current_round, timeout=timeout)
        # Bookkeeping
        self.evaluate_round(server_round=current_round, timeout=timeout)
        end_time = timeit.default_timer()
        elapsed = end_time - start_time
        log(INFO, "Federated PCA finished in %s", elapsed)
        return history

    def fit_round(
        self,
        server_round: int,
        timeout: Optional[float],
    ) -> Optional[Tuple[Optional[Parameters], Dict[str, Scalar], FitResultsAndFailures]]:
        # Get clients and their respective instructions.
        client_manager = self.client_manager()
        fit_ins = FitIns(self.parameters, {})
        clients = client_manager.sample(self.min_num_clients)
        client_instructions = [(client, fit_ins) for client in clients]

        if client_instructions == []:
            log(INFO, "No clients selected, cancel.")
            return None
        log(DEBUG, "Clients selected successfully.")

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

        # Merge clients' local principal components.
        client_eigenvalues = []
        client_pcs = []
        for _, fit_res in results:
            A = parameters_to_ndarrays(fit_res.parameters)
            pcs, eigenvalues = self.principal_components_packer.unpack(A)
            client_pcs.append(pcs)
            client_eigenvalues.append(eigenvalues)

        self.server_side_merger.set_pcs(client_pcs)
        self.server_side_merger.set_eigenvals(client_eigenvalues)
        self.server_side_merger.merge_subspaces()
        merged_pcs, merged_evals = self.server_side_merger.get_principal_components()

        parameters_aggregated = ndarrays_to_parameters(self.principal_components_packer.pack(merged_pcs, merged_evals))
        self.parameters = parameters_aggregated

        metrics_aggregated: Dict[str, Scalar] = {}
        return parameters_aggregated, metrics_aggregated, (results, failures)

    def evaluate_round(
        self,
        server_round: int,
        timeout: Optional[float],
    ) -> Optional[Tuple[Optional[float], Dict[str, Scalar], EvaluateResultsAndFailures]]:
        """Validate current global model on a number of clients."""

        # Get clients and their respective instructions from strategy
        client_manager = self.client_manager()
        fit_ins = EvaluateIns(self.parameters, {})
        clients = client_manager.sample(self.min_num_clients)
        client_instructions = [(client, fit_ins) for client in clients]

        if client_instructions == []:
            log(INFO, "No clients selected, cancel.")
            return None
        log(DEBUG, "Clients selected successfully.")

        if not client_instructions:
            log(INFO, "evaluate_round %s: no clients selected, cancel", server_round)
            return None
        log(
            DEBUG,
            "evaluate_round %s: strategy sampled %s clients (out of %s)",
            server_round,
            len(client_instructions),
            self._client_manager.num_available(),
        )

        # Collect `evaluate` results from all clients participating in this round
        results, failures = evaluate_clients(
            client_instructions,
            max_workers=self.max_workers,
            timeout=timeout,
        )
        log(
            DEBUG,
            "evaluate_round %s received %s results and %s failures",
            server_round,
            len(results),
            len(failures),
        )

        return None
