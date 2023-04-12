import concurrent.futures
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

        # Poll clients for sample counts
        log(INFO, "Polling Clients for sample counts")
        assert isinstance(self.strategy, ClientLevelDPFedAvgM)
        client_instructions = self.strategy.configure_poll(server_round=1, client_manager=self._client_manager)
        results, _ = poll_clients(
            client_instructions=client_instructions, max_workers=self.max_workers, timeout=timeout
        )

        sample_counts: List[int] = [int(result[1].properties["num_samples"]) for result in results]

        # If Weighted FedAvg, set sample counts to compute client weights
        if self.strategy.weighted_averaging:
            self.strategy.sample_counts = sample_counts

        return super().fit(num_rounds=num_rounds, timeout=timeout)


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
