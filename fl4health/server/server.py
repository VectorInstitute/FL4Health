import concurrent.futures
import timeit
from logging import INFO
from typing import List, Optional, Tuple, Union

from flwr.common.logger import log
from flwr.common.typing import Code, GetPropertiesIns, GetPropertiesRes
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.history import History
from flwr.server.server import Server

from fl4health.strategies.client_dp_fedavgm import ClientLevelDPFedAvgM

PollResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, GetPropertiesRes]],
    List[Union[Tuple[ClientProxy, GetPropertiesRes], BaseException]],
]


class ClientLevelDPWeightedFedAvgServer(Server):
    """
    Server to be used in case of Client Level Differential Privacy with weighted Federated Averaging.
    Modified the fit function to poll clients for sample counts prior to the first round of FL.
    """

    def __init__(self, *, client_manager: ClientManager, strategy: ClientLevelDPFedAvgM) -> None:
        super().__init__(client_manager=client_manager, strategy=strategy)

    def fit(self, num_rounds: int, timeout: Optional[float]) -> History:
        """Run federated averaging for a number of rounds."""
        history = History()

        # Poll clients for sample counts
        log(INFO, "Polling Clients for sample counts")
        assert isinstance(self.strategy, ClientLevelDPFedAvgM)
        client_instructions = self.strategy.configure_poll(server_round=1, client_manager=self._client_manager)
        results, _ = poll_clients(
            client_instructions=client_instructions, max_workers=self.max_workers, timeout=timeout
        )

        sample_counts: List[int] = [int(result[1].properties["num_samples"]) for result in results]

        # If Client Level DP and Weighted FedAvg, set sample counts to compute client weights
        if isinstance(self.strategy, ClientLevelDPFedAvgM) and self.strategy.weighted_averaging:
            self.strategy.sample_counts = sample_counts

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
            # Train model and replace previous global model
            res_fit = self.fit_round(server_round=current_round, timeout=timeout)
            if res_fit:
                parameters_prime, _, _ = res_fit  # fit_metrics_aggregated
                if parameters_prime:
                    self.parameters = parameters_prime

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
                history.add_metrics_centralized(server_round=current_round, metrics=metrics_cen)

            # Evaluate model on a sample of available clients
            res_fed = self.evaluate_round(server_round=current_round, timeout=timeout)
            if res_fed:
                loss_fed, evaluate_metrics_fed, _ = res_fed
                if loss_fed:
                    history.add_loss_distributed(server_round=current_round, loss=loss_fed)
                    history.add_metrics_distributed(server_round=current_round, metrics=evaluate_metrics_fed)

        # Bookkeeping
        end_time = timeit.default_timer()
        elapsed = end_time - start_time
        log(INFO, "FL finished in %s", elapsed)
        return history


def _handle_finished_future_after_poll(
    future: concurrent.futures.Future,
    results: List[Tuple[ClientProxy, GetPropertiesRes]],
    failures: List[Union[Tuple[ClientProxy, GetPropertiesRes], BaseException]],
) -> None:
    """Convert finished future into either a result or a failure for polling."""

    # Check if there was an exception
    failure = future.exception()
    if failure is not None:
        failures.append(failure)
        return

    # Successfully received a result from a client
    result: Tuple[ClientProxy, GetPropertiesRes] = future.result()
    _, res = result

    # Check result status code
    if res.status.code == Code.OK:
        results.append(result)
        return

    # Not successful, client returned a result where the status code is not OK
    failures.append(result)


def poll_client(client: ClientProxy, ins: GetPropertiesIns) -> Tuple[ClientProxy, GetPropertiesRes]:
    """Get Properties of client"""
    property_res: GetPropertiesRes = client.get_properties(ins=ins, timeout=None)
    return client, property_res


def poll_clients(
    client_instructions: List[Tuple[ClientProxy, GetPropertiesIns]],
    max_workers: Optional[int],
    timeout: Optional[float],
) -> PollResultsAndFailures:
    """Poll clients concurrently on all selected clients."""

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        submitted_fs = {
            executor.submit(poll_client, client_proxy, property_ins)
            for client_proxy, property_ins in client_instructions
        }
        finished_fs, _ = concurrent.futures.wait(
            fs=submitted_fs,
            timeout=None,  # Handled in the respective communication stack
        )

    # Gather results
    results: List[Tuple[ClientProxy, GetPropertiesRes]] = []
    failures: List[Union[Tuple[ClientProxy, GetPropertiesRes], BaseException]] = []
    for future in finished_fs:
        _handle_finished_future_after_poll(future=future, results=results, failures=failures)

    return results, failures
