import concurrent.futures
from typing import List, Optional, Tuple, Union

from flwr.common.typing import Code, GetPropertiesIns, GetPropertiesRes
from flwr.server.client_proxy import ClientProxy

PollResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, GetPropertiesRes]],
    List[Union[Tuple[ClientProxy, GetPropertiesRes], BaseException]],
]


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
